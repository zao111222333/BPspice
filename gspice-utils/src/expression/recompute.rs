use super::{
    op::{BinaryOp, CmpMethod, CmpOp, Cond, Powf, UnaryOp},
    Expression, Op, ScalarTensor, Tensor,
};
use itertools::izip;
use num_traits::Zero;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[cfg(test)]
pub(crate) static TEST_RECOMPUTE_COUNT: AtomicUsize = AtomicUsize::new(0);

impl Expression {
    pub(super) fn recompute<'a>(&'a self) -> RecomputeScalarTensor<'a> {
        #[cfg(test)]
        {
            TEST_RECOMPUTE_COUNT.fetch_add(1, Relaxed);
        }
        match self {
            Expression::Const(f) => RecomputeScalarTensor::Scalar(f),
            Expression::Tensor(tensor) => match tensor.change_marker().change_state() {
                ChangeState::Changed => RecomputeScalarTensor::TensorChanged(tensor),
                ChangeState::NoChange => RecomputeScalarTensor::TensorNoChange(tensor),
                ChangeState::NeedSearch => match tensor.op() {
                    Op::Assgin => RecomputeScalarTensor::nochange(tensor),
                    Op::Powf(node, n) => Powf::recompute(*n, node, tensor),
                    Op::Cond(cond, on_true, on_false) => {
                        Cond::recompute(cond, on_true, on_false, tensor)
                    }
                    Op::Unary(node, unary_op) => unary_op.recompute(node, tensor),
                    Op::Binary(lhs, rhs, binary_op) => binary_op.recompute(lhs, rhs, tensor),
                    Op::Cmp(lhs, rhs, cmp_op, cmp_method) => {
                        cmp_op.recompute(lhs, rhs, cmp_method, tensor)
                    }
                },
            },
        }
    }
}

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
#[pyfunction]
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
    #[rustfmt::skip]
    fn recompute<'a>(
        &self,
        lhs: &Expression,
        rhs: &Expression,
        tensor: &'a Tensor,
    ) -> RecomputeScalarTensor<'a> {
        let [fn_forward_lhs_rhs, fn_forward_rhs_lhs] = self.forward();
        match (lhs.recompute(), rhs.recompute()) {
            (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::Scalar(_))
                => unreachable!(),
            (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorNoChange(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::Scalar(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::TensorNoChange(_))
                => RecomputeScalarTensor::nochange(tensor),
            (RecomputeScalarTensor::Scalar(lhs_x), RecomputeScalarTensor::TensorChanged(rhs_tensor))
                => RecomputeScalarTensor::change(
                    tensor,
                    rhs_tensor.broadcast_iter_binary_op(*lhs_x, fn_forward_rhs_lhs),
                ),
            (RecomputeScalarTensor::TensorChanged(lhs_tensor), RecomputeScalarTensor::Scalar(rhs_x))
                => RecomputeScalarTensor::change(
                    tensor,
                    lhs_tensor.broadcast_iter_binary_op(*rhs_x, fn_forward_lhs_rhs),
                ),
            (RecomputeScalarTensor::TensorChanged(lhs_tensor), RecomputeScalarTensor::TensorNoChange(rhs_tensor))
            | (RecomputeScalarTensor::TensorChanged(lhs_tensor), RecomputeScalarTensor::TensorChanged(rhs_tensor))
            | (RecomputeScalarTensor::TensorNoChange(lhs_tensor), RecomputeScalarTensor::TensorChanged(rhs_tensor))
                => RecomputeScalarTensor::change(
                    tensor,
                    lhs_tensor.iter_binary_op(rhs_tensor, fn_forward_lhs_rhs),
                ),
        }
    }
}

impl CmpOp {
    fn recompute<'a>(
        &self,
        lhs: &Expression,
        rhs: &Expression,
        cmp_method: &CmpMethod,
        tensor: &'a Tensor,
    ) -> RecomputeScalarTensor<'a> {
        match (lhs.recompute(), rhs.recompute()) {
            (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::Scalar(_)) => unreachable!(),
            (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorNoChange(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::Scalar(_))
            | (
                RecomputeScalarTensor::TensorNoChange(_),
                RecomputeScalarTensor::TensorNoChange(_),
            ) => RecomputeScalarTensor::nochange(tensor),
            (
                RecomputeScalarTensor::Scalar(lhs_x),
                RecomputeScalarTensor::TensorChanged(rhs_tensor),
            ) => RecomputeScalarTensor::change(
                tensor,
                self.forward_iter_fix_lhs(
                    cmp_method,
                    *lhs_x,
                    rhs_tensor.values().read().unwrap().iter(),
                ),
            ),
            (
                RecomputeScalarTensor::TensorChanged(lhs_tensor),
                RecomputeScalarTensor::Scalar(rhs_x),
            ) => RecomputeScalarTensor::change(
                tensor,
                self.forward_iter_fix_rhs(
                    cmp_method,
                    *rhs_x,
                    lhs_tensor.values().read().unwrap().iter(),
                ),
            ),
            (
                RecomputeScalarTensor::TensorChanged(lhs_tensor),
                RecomputeScalarTensor::TensorNoChange(rhs_tensor),
            )
            | (
                RecomputeScalarTensor::TensorChanged(lhs_tensor),
                RecomputeScalarTensor::TensorChanged(rhs_tensor),
            )
            | (
                RecomputeScalarTensor::TensorNoChange(lhs_tensor),
                RecomputeScalarTensor::TensorChanged(rhs_tensor),
            ) => RecomputeScalarTensor::change(
                tensor,
                self.forward_iter(
                    cmp_method,
                    izip!(
                        lhs_tensor.values().read().unwrap().iter(),
                        rhs_tensor.values().read().unwrap().iter()
                    ),
                ),
            ),
        }
    }
}

impl Powf {
    fn recompute<'a>(n: f64, node: &Expression, tensor: &'a Tensor) -> RecomputeScalarTensor<'a> {
        match node.recompute() {
            RecomputeScalarTensor::Scalar(_) => unreachable!(),
            RecomputeScalarTensor::TensorNoChange(_) => RecomputeScalarTensor::nochange(tensor),
            RecomputeScalarTensor::TensorChanged(node_tensor) => RecomputeScalarTensor::change(
                tensor,
                node_tensor.broadcast_iter_binary_op(n, Powf::forward),
            ),
        }
    }
}

impl Cond {
    #[rustfmt::skip]
    fn recompute<'a>(
        cond: &Expression,
        on_true: &Expression,
        on_false: &Expression,
        tensor: &'a Tensor,
    ) -> RecomputeScalarTensor<'a> {
        match (cond.recompute(), on_true.recompute(), on_false.recompute()){
            (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::Scalar(_))
                => unreachable!(),
            (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorNoChange(_))
            | (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::Scalar(_))
            | (RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::TensorNoChange(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::Scalar(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorNoChange(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::Scalar(_))
            | (RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::TensorNoChange(_))
                => RecomputeScalarTensor::nochange(tensor),
            (RecomputeScalarTensor::Scalar(cond_x), RecomputeScalarTensor::Scalar(_), RecomputeScalarTensor::TensorChanged(on_false_tensor))
            | (RecomputeScalarTensor::Scalar(cond_x), RecomputeScalarTensor::TensorNoChange(_), RecomputeScalarTensor::TensorChanged(on_false_tensor))
                => if cond_x.is_zero() {
                    RecomputeScalarTensor::change(tensor, on_false_tensor.values().read().unwrap().clone())
                } else {
                    RecomputeScalarTensor::nochange(tensor)
                },
            (RecomputeScalarTensor::Scalar(cond_x), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::Scalar(_))
            | (RecomputeScalarTensor::Scalar(cond_x), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::TensorNoChange(_))
                => if cond_x.is_zero() {
                    RecomputeScalarTensor::nochange(tensor)
                } else {
                    RecomputeScalarTensor::change(tensor, on_true_tensor.values().read().unwrap().clone())
                },
            (RecomputeScalarTensor::Scalar(cond_x), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::TensorChanged(on_false_tensor))
                => if cond_x.is_zero() {
                    RecomputeScalarTensor::change(tensor, on_false_tensor.values().read().unwrap().clone())
                } else {
                    RecomputeScalarTensor::change(tensor, on_true_tensor.values().read().unwrap().clone())
                },
            (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::Scalar(on_true_x), RecomputeScalarTensor::Scalar(on_false_x))
                => RecomputeScalarTensor::change(tensor, Self::iter_tensor_x_x(cond_tensor, *on_true_x, *on_false_x)),
            (RecomputeScalarTensor::TensorNoChange(cond_tensor), RecomputeScalarTensor::Scalar(on_true_x), RecomputeScalarTensor::TensorChanged(on_false_tensor))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::Scalar(on_true_x), RecomputeScalarTensor::TensorNoChange(on_false_tensor))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::Scalar(on_true_x), RecomputeScalarTensor::TensorChanged(on_false_tensor))
                => RecomputeScalarTensor::change(tensor, Self::iter_tensor_x_tensor(cond_tensor, *on_true_x, on_false_tensor)),
            (RecomputeScalarTensor::TensorNoChange(cond_tensor), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::Scalar(on_false_x))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::TensorNoChange(on_true_tensor), RecomputeScalarTensor::Scalar(on_false_x))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::Scalar(on_false_x))
                => RecomputeScalarTensor::change(tensor, Self::iter_tensor_tensor_x(cond_tensor, on_true_tensor, *on_false_x)),
            (RecomputeScalarTensor::TensorNoChange(cond_tensor), RecomputeScalarTensor::TensorNoChange(on_true_tensor), RecomputeScalarTensor::TensorChanged(on_false_tensor))
            | (RecomputeScalarTensor::TensorNoChange(cond_tensor), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::TensorNoChange(on_false_tensor))
            | (RecomputeScalarTensor::TensorNoChange(cond_tensor), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::TensorChanged(on_false_tensor))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::TensorNoChange(on_true_tensor), RecomputeScalarTensor::TensorNoChange(on_false_tensor))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::TensorNoChange(on_true_tensor), RecomputeScalarTensor::TensorChanged(on_false_tensor))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::TensorNoChange(on_false_tensor))
            | (RecomputeScalarTensor::TensorChanged(cond_tensor), RecomputeScalarTensor::TensorChanged(on_true_tensor), RecomputeScalarTensor::TensorChanged(on_false_tensor))
                => RecomputeScalarTensor::change(tensor, Self::iter_tensor_tensor_tensor(cond_tensor, on_true_tensor, on_false_tensor)),
        }
    }
}

impl UnaryOp {
    fn recompute<'a>(&self, node: &Expression, tensor: &'a Tensor) -> RecomputeScalarTensor<'a> {
        match node.recompute() {
            RecomputeScalarTensor::Scalar(_) => unreachable!(),
            RecomputeScalarTensor::TensorNoChange(_) => RecomputeScalarTensor::nochange(tensor),
            RecomputeScalarTensor::TensorChanged(node_tensor) => {
                RecomputeScalarTensor::change(tensor, node_tensor.iter_unary_op(self.forward()))
            }
        }
    }
}
