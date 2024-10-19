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

pub enum RecomputeScalarTensor<'a> {
    Scalar(&'a f64),
    TensorNoChange(&'a Tensor),
    TensorChanged(&'a Tensor),
}

impl<'a> From<RecomputeScalarTensor<'a>> for ScalarTensor<'a> {
    fn from(value: RecomputeScalarTensor<'a>) -> Self {
        match value {
            RecomputeScalarTensor::Scalar(f) => ScalarTensor::Scalar(f),
            RecomputeScalarTensor::TensorNoChange(tensor)
            | RecomputeScalarTensor::TensorChanged(tensor) => ScalarTensor::Tensor(tensor),
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
pub struct ChangeMarker(AtomicUsize);
impl ChangeMarker {
    pub fn recompute_done() {
        // No need async, use Relaxed
        COUNTER.fetch_add(2, Relaxed);
    }
    pub(super) const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }
    pub(super) fn mark_searched_change(&self) {
        self.0.store(COUNTER.load(Relaxed) + 1, Relaxed);
    }
    fn mark_searched_nochange(&self) {
        self.0.store(COUNTER.load(Relaxed) + 2, Relaxed);
    }
    fn parameter_search_change(&self) -> bool {
        let counter = COUNTER.load(Relaxed);
        if self.0.load(Relaxed) == counter + 1 {
            true
        } else {
            self.0.store(counter + 2, Relaxed);
            false
        }
    }
    fn op_state(&self) -> ChangeState {
        let counter = COUNTER.load(Relaxed);
        match counter + 2 - self.0.load(Relaxed) {
            1 => ChangeState::Changed,
            0 => ChangeState::NoChange,
            _ => ChangeState::NeedSearch,
        }
    }
}

impl BinaryOp {
    pub(super) fn recompute<'a>(
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
    pub(super) fn recompute<'a>(
        &self,
        node: &Expression,
        tensor: &'a Tensor,
    ) -> RecomputeScalarTensor<'a> {
        match node.recompute() {
            RecomputeScalarTensor::Scalar(_) => todo!(),
            RecomputeScalarTensor::TensorNoChange(_) => todo!(),
            RecomputeScalarTensor::TensorChanged(_) => todo!(),
        }
    }
}

impl Expression {
    /// return if changed inside
    pub fn recompute<'a>(&'a self) -> RecomputeScalarTensor<'a> {
        match self {
            Expression::Const(f) => RecomputeScalarTensor::Scalar(f),
            Expression::Parameter(tensor) => {
                if tensor.change_marker().parameter_search_change() {
                    RecomputeScalarTensor::TensorChanged(tensor)
                } else {
                    RecomputeScalarTensor::TensorNoChange(tensor)
                }
            }
            Expression::Operation(tensor, op) => match tensor.change_marker().op_state() {
                ChangeState::Changed => RecomputeScalarTensor::TensorChanged(tensor),
                ChangeState::NoChange => RecomputeScalarTensor::TensorNoChange(tensor),
                ChangeState::NeedSearch => match op.as_ref() {
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

#[test]
fn test_recompute() {
    use std::ops::*;
    let const1 = Expression::constant(1.0);
    let const2 = Expression::constant(-1.0);
    let (param1, param1_tensor) = Expression::parameter(vec![1.0, 2.0, 3.0], true);
    let (param2, param2_tensor) = Expression::parameter(vec![-1.0, -2.0, -3.0], true);

    let exp1 = param1.add(&param2);
    // let exp1 = &param1 + &param2;
    let exp2 = param1.add(&const1);
    let exp3 = param1.sub(&const1);
    let exp4 = const1.sub(&param1);
    println!("param1 {}", param1);
    println!("param2 {}", param2);
    // println!("exp1 {:?}", exp1);
    println!("exp1 {}", exp1);
    // println!("exp2 {:?}", exp2);
    println!("exp2 {}", exp2);
    println!("exp3 {}", exp3);
    println!("exp4 {}", exp4);
    println!("const1 {}", const1);
    param1_tensor.update(vec![0.0, 0.0, 0.0]);
    println!("param1 {}", param1);
    exp4.recompute();
    exp1.recompute();
    println!("exp4 {:?}", exp4);
    println!("exp1 {:?}", exp1);
    ChangeMarker::recompute_done();
}
