use std::sync::{Arc, RwLock};

use super::{ChangeMarker, Expression, GradId, Tensor};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryOp {
    Neg,
    Sin,
    Cos,
    Tanh,
    Tan,
    Ceil,
    Floor,
    Round,
    Sign,
    Sqrt,
    Sqr,
    Log,
    Exp,
    Abs,
    Erf,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}
trait BinaryOpT {
    const OP: BinaryOp;
    fn op_fn(lhs: f64, rhs: f64) -> f64;
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64;
}

struct Add;
impl BinaryOpT for Add {
    const OP: BinaryOp = BinaryOp::Add;
    fn op_fn(lhs: f64, rhs: f64) -> f64 {
        lhs + rhs
    }
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64 {
        rhs + lhs
    }
}
impl<'a, 'b> core::ops::Add<&'b Expression> for &'a Expression {
    type Output = Expression;
    fn add(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Add>(rhs)
    }
}

struct Sub;
impl BinaryOpT for Sub {
    const OP: BinaryOp = BinaryOp::Sub;
    fn op_fn(lhs: f64, rhs: f64) -> f64 {
        lhs - rhs
    }
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64 {
        rhs - lhs
    }
}
impl<'a, 'b> core::ops::Sub<&'b Expression> for &'a Expression {
    type Output = Expression;
    fn sub(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Sub>(rhs)
    }
}

struct Mul;
impl BinaryOpT for Mul {
    const OP: BinaryOp = BinaryOp::Mul;
    fn op_fn(lhs: f64, rhs: f64) -> f64 {
        lhs * rhs
    }
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64 {
        rhs * lhs
    }
}
impl<'a, 'b> core::ops::Mul<&'b Expression> for &'a Expression {
    type Output = Expression;
    fn mul(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Mul>(rhs)
    }
}

struct Div;
impl BinaryOpT for Div {
    const OP: BinaryOp = BinaryOp::Div;
    fn op_fn(lhs: f64, rhs: f64) -> f64 {
        lhs / rhs
    }
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64 {
        rhs / lhs
    }
}
impl<'a, 'b> core::ops::Div<&'b Expression> for &'a Expression {
    type Output = Expression;
    fn div(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Div>(rhs)
    }
}

struct Pow;
impl BinaryOpT for Pow {
    const OP: BinaryOp = BinaryOp::Pow;
    fn op_fn(lhs: f64, rhs: f64) -> f64 {
        lhs.powf(rhs)
    }
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64 {
        rhs.powf(lhs)
    }
}

struct Min;
impl BinaryOpT for Min {
    const OP: BinaryOp = BinaryOp::Min;
    fn op_fn(lhs: f64, rhs: f64) -> f64 {
        lhs.min(rhs)
    }
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64 {
        rhs.min(lhs)
    }
}
struct Max;
impl BinaryOpT for Max {
    const OP: BinaryOp = BinaryOp::Max;
    fn op_fn(lhs: f64, rhs: f64) -> f64 {
        lhs.max(rhs)
    }
    fn op_fn_inverse(lhs: f64, rhs: f64) -> f64 {
        rhs.max(lhs)
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    // Copy(Expression),
    Powf(Expression, f64),
    // (op1 cmp_op op2)? op3 : op4
    Cond(Expression, CmpOp, Expression, Expression, Expression),
    Cmp(Expression, CmpOp),
    Unary(Expression, UnaryOp),
    Binary(Expression, Expression, BinaryOp),
}

impl BinaryOp {
    pub(super) const fn op_fn(&self) -> fn(f64, f64) -> f64 {
        match self {
            BinaryOp::Add => Add::op_fn,
            BinaryOp::Sub => Sub::op_fn,
            BinaryOp::Mul => Mul::op_fn,
            BinaryOp::Div => Div::op_fn,
            BinaryOp::Pow => Pow::op_fn,
            BinaryOp::Min => Min::op_fn,
            BinaryOp::Max => Max::op_fn,
        }
    }
    pub(super) const fn op_fn_inverse(&self) -> fn(f64, f64) -> f64 {
        match self {
            BinaryOp::Add => Add::op_fn_inverse,
            BinaryOp::Sub => Sub::op_fn_inverse,
            BinaryOp::Mul => Mul::op_fn_inverse,
            BinaryOp::Div => Div::op_fn_inverse,
            BinaryOp::Pow => Pow::op_fn_inverse,
            BinaryOp::Min => Min::op_fn_inverse,
            BinaryOp::Max => Max::op_fn_inverse,
        }
    }
}

impl Tensor {
    pub(super) fn iter_binary_op(&self, rhs: &Self, op_fn: fn(f64, f64) -> f64) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .zip(rhs.values().read().unwrap().iter())
            .map(|(v1, v2)| op_fn(*v1, *v2))
            .collect()
    }
    pub(super) fn broadcast_iter_binary_op(
        &self,
        rhs: f64,
        op_fn: fn(f64, f64) -> f64,
    ) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .map(|v| op_fn(*v, rhs))
            .collect()
    }
    pub(super) fn binary_op(&self, rhs: &Self, op_fn: fn(f64, f64) -> f64) -> Self {
        Self(Arc::new((
            if self.grad_id().is_some() || rhs.grad_id().is_some() {
                Some(GradId::new())
            } else {
                None
            },
            RwLock::new(self.iter_binary_op(rhs, op_fn)),
            ChangeMarker::new(),
        )))
    }
    pub(super) fn broadcast_binary_op(&self, rhs: f64, op_fn: fn(f64, f64) -> f64) -> Self {
        Self(Arc::new((
            if self.grad_id().is_some() {
                Some(GradId::new())
            } else {
                None
            },
            RwLock::new(self.broadcast_iter_binary_op(rhs, op_fn)),
            ChangeMarker::new(),
        )))
    }
}

impl Expression {
    pub fn pow(&self, rhs: &Self) -> Self {
        self.binary_op::<Pow>(rhs)
    }
    pub fn min(&self, rhs: &Self) -> Self {
        self.binary_op::<Min>(rhs)
    }
    pub fn max(&self, rhs: &Self) -> Self {
        self.binary_op::<Max>(rhs)
    }
    fn binary_op<T: BinaryOpT>(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Self::Const(v1), Self::Const(v2)) => Self::Const(T::op_fn(*v1, *v2)),
            (Self::Const(v1), Self::Parameter(tensor2)) => Self::Operation(
                tensor2.broadcast_binary_op(*v1, T::op_fn_inverse),
                Arc::new(Op::Binary(
                    Self::Const(*v1),
                    Self::Parameter(tensor2.clone()),
                    T::OP,
                )),
            ),
            (Self::Parameter(tensor1), Self::Const(v2)) => Self::Operation(
                tensor1.broadcast_binary_op(*v2, T::op_fn),
                Arc::new(Op::Binary(
                    Self::Parameter(tensor1.clone()),
                    Self::Const(*v2),
                    T::OP,
                )),
            ),
            (Self::Const(v1), Self::Operation(tensor2, op2)) => Self::Operation(
                tensor2.broadcast_binary_op(*v1, T::op_fn_inverse),
                Arc::new(Op::Binary(
                    Self::Const(*v1),
                    Self::Operation(tensor2.clone(), op2.clone()),
                    T::OP,
                )),
            ),
            (Self::Operation(tensor1, op1), Self::Const(v2)) => Self::Operation(
                tensor1.broadcast_binary_op(*v2, T::op_fn),
                Arc::new(Op::Binary(
                    Self::Operation(tensor1.clone(), op1.clone()),
                    Self::Const(*v2),
                    T::OP,
                )),
            ),
            (Expression::Parameter(tensor1), Expression::Parameter(tensor2)) => Self::Operation(
                tensor1.binary_op(tensor2, T::op_fn),
                Arc::new(Op::Binary(
                    Self::Parameter(tensor1.clone()),
                    Self::Parameter(tensor2.clone()),
                    T::OP,
                )),
            ),
            (Expression::Parameter(tensor1), Expression::Operation(tensor2, op2)) => {
                Self::Operation(
                    tensor1.binary_op(tensor2, T::op_fn),
                    Arc::new(Op::Binary(
                        Self::Parameter(tensor1.clone()),
                        Self::Operation(tensor2.clone(), op2.clone()),
                        T::OP,
                    )),
                )
            }
            (Expression::Operation(tensor1, op1), Expression::Parameter(tensor2)) => {
                Self::Operation(
                    tensor1.binary_op(tensor2, T::op_fn),
                    Arc::new(Op::Binary(
                        Self::Operation(tensor1.clone(), op1.clone()),
                        Self::Parameter(tensor2.clone()),
                        T::OP,
                    )),
                )
            }
            (Expression::Operation(tensor1, op1), Expression::Operation(tensor2, op2)) => {
                Self::Operation(
                    tensor1.binary_op(tensor2, T::op_fn),
                    Arc::new(Op::Binary(
                        Self::Operation(tensor1.clone(), op1.clone()),
                        Self::Operation(tensor2.clone(), op2.clone()),
                        T::OP,
                    )),
                )
            }
        }
    }
}
