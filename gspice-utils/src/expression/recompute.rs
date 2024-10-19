use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use super::{
    op::{BinaryOp, UnaryOp},
    Expression, Op, ScalarTensor, Tensor,
};

enum ChangeState {
    NeedSearch,
    Changed,
    NoChange,
}

pub(super) enum RecomputeScalarTensor<'a> {
    Scalar(&'a f64),
    TensorNoChange(&'a Tensor),
    TensorChanged(&'a Tensor),
}

impl<'a> From<RecomputeScalarTensor<'a>> for ScalarTensor<'a> {
    fn from(value: RecomputeScalarTensor<'a>) -> Self {
        match value {
            RecomputeScalarTensor::Scalar(f) => ScalarTensor::Scalar(f),
            RecomputeScalarTensor::TensorNoChange(tensor)
            | RecomputeScalarTensor::TensorChanged(tensor) => ScalarTensor::Tensor(tensor.values()),
        }
    }
}

impl<'a> RecomputeScalarTensor<'a> {
    fn change(tensor: &'a Tensor, values: Vec<f64>) -> Self {
        let mut write = tensor.values().write().unwrap();
        *write = values;
        tensor.change_marker().mark_searched_change();
        RecomputeScalarTensor::TensorChanged(tensor)
    }
    fn nochange(tensor: &'a Tensor) -> Self {
        tensor.change_marker().mark_searched_nochange();
        Self::TensorNoChange(tensor)
    }
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn before_update() {
    // No need async, use Relaxed
    COUNTER.fetch_add(2, Relaxed);
}

/// When ChangeMarker::COUNTER is 2n,
///
/// 2n-1 , 2n : have not been searched
///
/// 2n+1 : searched, change
///
/// 2n+2 : searched, no change inside
///
/// update tensor makes its marker become 2n+1
#[derive(Debug)]
pub(crate) struct ChangeMarker(AtomicUsize);
impl ChangeMarker {
    pub(super) const fn new() -> Self {
        Self(AtomicUsize::new(2))
    }
    pub(super) fn mark_searched_change(&self) {
        self.0.store(COUNTER.load(Relaxed) + 1, Relaxed);
    }
    fn mark_searched_nochange(&self) {
        self.0.store(COUNTER.load(Relaxed) + 2, Relaxed);
    }
    fn change_state(&self) -> ChangeState {
        let counter = COUNTER.load(Relaxed);
        match counter + 2 - self.0.load(Relaxed) {
            1 => ChangeState::Changed,
            0 => ChangeState::NoChange,
            _ => ChangeState::NeedSearch,
        }
    }
}

impl BinaryOp {
    fn recompute<'a>(
        &self,
        lhs: &Expression,
        rhs: &Expression,
        tensor: &'a Tensor,
    ) -> RecomputeScalarTensor<'a> {
        match (lhs.recompute(), rhs.recompute()) {
            (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::Scalar(_))
            | (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorNoChange(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::Scalar(_))
            | (
                RecomputeScalarTensor::TensorNoChange(_),
                RecomputeScalarTensor::TensorNoChange(_),
            ) => RecomputeScalarTensor::nochange(tensor),
            (RecomputeScalarTensor::Scalar(lhs_v), RecomputeScalarTensor::TensorChanged(rhs_v)) => {
                RecomputeScalarTensor::change(
                    tensor,
                    rhs_v.broadcast_iter_binary_op(*lhs_v, self.op_fn_inverse()),
                )
            }
            (RecomputeScalarTensor::TensorChanged(lhs_v), RecomputeScalarTensor::Scalar(rhs_v)) => {
                RecomputeScalarTensor::change(
                    tensor,
                    lhs_v.broadcast_iter_binary_op(*rhs_v, self.op_fn()),
                )
            }
            (
                RecomputeScalarTensor::TensorChanged(lhs_v),
                RecomputeScalarTensor::TensorNoChange(rhs_v),
            )
            | (
                RecomputeScalarTensor::TensorChanged(lhs_v),
                RecomputeScalarTensor::TensorChanged(rhs_v),
            )
            | (
                RecomputeScalarTensor::TensorNoChange(lhs_v),
                RecomputeScalarTensor::TensorChanged(rhs_v),
            ) => RecomputeScalarTensor::change(tensor, lhs_v.iter_binary_op(rhs_v, self.op_fn())),
        }
    }
}

impl UnaryOp {
    fn recompute<'a>(&self, node: &Expression, tensor: &'a Tensor) -> RecomputeScalarTensor<'a> {
        match node.recompute() {
            RecomputeScalarTensor::Scalar(_) => unreachable!(),
            RecomputeScalarTensor::TensorNoChange(_) => RecomputeScalarTensor::nochange(tensor),
            RecomputeScalarTensor::TensorChanged(node) => {
                RecomputeScalarTensor::change(tensor, node.iter_unary_op(self.op_fn()))
            }
        }
    }
}

impl Expression {
    pub(super) fn recompute<'a>(&'a self) -> RecomputeScalarTensor<'a> {
        match self {
            Expression::Const(f) => RecomputeScalarTensor::Scalar(f),
            Expression::Tensor(tensor) => match tensor.change_marker().change_state() {
                ChangeState::Changed => RecomputeScalarTensor::TensorChanged(tensor),
                ChangeState::NoChange => RecomputeScalarTensor::TensorNoChange(tensor),
                ChangeState::NeedSearch => match tensor.op() {
                    Op::Assgin => RecomputeScalarTensor::nochange(tensor),
                    Op::Powf(expression, _) => todo!(),
                    Op::Cond(expression, cmp_op, expression1, expression2, expression3) => {
                        todo!()
                    }
                    Op::Cmp(expression, cmp_op) => todo!(),
                    Op::Unary(node, unary_op) => unary_op.recompute(node, tensor),
                    Op::Binary(lhs, rhs, binary_op) => binary_op.recompute(lhs, rhs, tensor),
                },
            },
        }
    }
}