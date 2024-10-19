use std::sync::{Arc, RwLock};

use super::{ChangeMarker, Expression, GradId, Tensor};

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   UnaryOp   ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
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

trait UnaryOpT {
    const OP: UnaryOp;
    fn op_fn(x: f64) -> f64;
}

struct Neg;
impl UnaryOpT for Neg {
    const OP: UnaryOp = UnaryOp::Neg;
    fn op_fn(x: f64) -> f64 {
        -x
    }
}
impl<'a> core::ops::Neg for &'a Expression {
    type Output = Expression;

    fn neg(self) -> Self::Output {
        Expression::unary_op::<Neg>(&self)
    }
}

struct Sin;
impl UnaryOpT for Sin {
    const OP: UnaryOp = UnaryOp::Sin;
    fn op_fn(x: f64) -> f64 {
        x.sin()
    }
}
struct Cos;
impl UnaryOpT for Cos {
    const OP: UnaryOp = UnaryOp::Cos;
    fn op_fn(x: f64) -> f64 {
        x.cos()
    }
}
struct Tanh;
impl UnaryOpT for Tanh {
    const OP: UnaryOp = UnaryOp::Tanh;
    fn op_fn(x: f64) -> f64 {
        x.tanh()
    }
}
struct Tan;
impl UnaryOpT for Tan {
    const OP: UnaryOp = UnaryOp::Tan;
    fn op_fn(x: f64) -> f64 {
        x.tan()
    }
}
struct Ceil;
impl UnaryOpT for Ceil {
    const OP: UnaryOp = UnaryOp::Ceil;
    fn op_fn(x: f64) -> f64 {
        x.ceil()
    }
}
struct Floor;
impl UnaryOpT for Floor {
    const OP: UnaryOp = UnaryOp::Floor;
    fn op_fn(x: f64) -> f64 {
        x.floor()
    }
}
struct Round;
impl UnaryOpT for Round {
    const OP: UnaryOp = UnaryOp::Round;
    fn op_fn(x: f64) -> f64 {
        x.round()
    }
}
struct Sign;
impl UnaryOpT for Sign {
    const OP: UnaryOp = UnaryOp::Sign;
    fn op_fn(x: f64) -> f64 {
        x.signum()
    }
}
struct Sqrt;
impl UnaryOpT for Sqrt {
    const OP: UnaryOp = UnaryOp::Sqrt;
    fn op_fn(x: f64) -> f64 {
        x.sqrt()
    }
}
struct Sqr;
impl UnaryOpT for Sqr {
    const OP: UnaryOp = UnaryOp::Sqr;
    fn op_fn(x: f64) -> f64 {
        x.powi(2)
    }
}
struct Log;
impl UnaryOpT for Log {
    const OP: UnaryOp = UnaryOp::Log;
    fn op_fn(x: f64) -> f64 {
        x.ln()
    }
}
struct Exp;
impl UnaryOpT for Exp {
    const OP: UnaryOp = UnaryOp::Exp;
    fn op_fn(x: f64) -> f64 {
        x.exp()
    }
}
struct Abs;
impl UnaryOpT for Abs {
    const OP: UnaryOp = UnaryOp::Abs;
    fn op_fn(x: f64) -> f64 {
        x.abs()
    }
}
struct Erf;
impl UnaryOpT for Erf {
    const OP: UnaryOp = UnaryOp::Erf;
    fn op_fn(x: f64) -> f64 {
        candle_core::cpu::erf::erf(x)
    }
}

impl UnaryOp {
    pub(super) const fn op_fn(&self) -> fn(f64) -> f64 {
        match self {
            Self::Neg => Neg::op_fn,
            Self::Sin => Sin::op_fn,
            Self::Cos => Cos::op_fn,
            Self::Tanh => Tanh::op_fn,
            Self::Tan => Tan::op_fn,
            Self::Ceil => Ceil::op_fn,
            Self::Floor => Floor::op_fn,
            Self::Round => Round::op_fn,
            Self::Sign => Sign::op_fn,
            Self::Sqrt => Sqrt::op_fn,
            Self::Sqr => Sqr::op_fn,
            Self::Log => Log::op_fn,
            Self::Exp => Exp::op_fn,
            Self::Abs => Abs::op_fn,
            Self::Erf => Erf::op_fn,
        }
    }
}

impl Tensor {
    pub(super) fn iter_unary_op(&self, op_fn: fn(f64) -> f64) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .map(|x| op_fn(*x))
            .collect()
    }
    pub(super) fn unary_op(&self, op_fn: fn(f64) -> f64) -> Self {
        Self(Arc::new((
            if self.grad_id().is_some() {
                Some(GradId::new())
            } else {
                None
            },
            RwLock::new(self.iter_unary_op(op_fn)),
            ChangeMarker::new(),
        )))
    }
}

impl Expression {
    pub fn sin(&self) -> Self {
        Expression::unary_op::<Sin>(&self)
    }
    pub fn cos(&self) -> Self {
        Expression::unary_op::<Cos>(&self)
    }
    pub fn tanh(&self) -> Self {
        Expression::unary_op::<Tanh>(&self)
    }
    pub fn tan(&self) -> Self {
        Expression::unary_op::<Tan>(&self)
    }
    pub fn ceil(&self) -> Self {
        Expression::unary_op::<Ceil>(&self)
    }
    pub fn floor(&self) -> Self {
        Expression::unary_op::<Floor>(&self)
    }
    pub fn round(&self) -> Self {
        Expression::unary_op::<Round>(&self)
    }
    pub fn sign(&self) -> Self {
        Expression::unary_op::<Sign>(&self)
    }
    pub fn sqrt(&self) -> Self {
        Expression::unary_op::<Sqrt>(&self)
    }
    pub fn sqr(&self) -> Self {
        Expression::unary_op::<Sqr>(&self)
    }
    pub fn log(&self) -> Self {
        Expression::unary_op::<Log>(&self)
    }
    pub fn exp(&self) -> Self {
        Expression::unary_op::<Exp>(&self)
    }
    pub fn abs(&self) -> Self {
        Expression::unary_op::<Abs>(&self)
    }
    pub fn erf(&self) -> Self {
        Expression::unary_op::<Erf>(&self)
    }
    fn unary_op<T: UnaryOpT>(&self) -> Self {
        match self {
            Expression::Const(x) => Expression::Const(T::op_fn(*x)),
            Expression::Parameter(tensor) => Expression::Operation(
                tensor.unary_op(T::op_fn),
                Arc::new(Op::Unary(Self::Parameter(tensor.clone()), T::OP)),
            ),
            Expression::Operation(tensor, op) => Expression::Operation(
                tensor.unary_op(T::op_fn),
                Arc::new(Op::Unary(
                    Self::Operation(tensor.clone(), op.clone()),
                    T::OP,
                )),
            ),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   BinaryOp   ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

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

impl BinaryOp {
    pub(super) const fn op_fn(&self) -> fn(f64, f64) -> f64 {
        match self {
            Self::Add => Add::op_fn,
            Self::Sub => Sub::op_fn,
            Self::Mul => Mul::op_fn,
            Self::Div => Div::op_fn,
            Self::Pow => Pow::op_fn,
            Self::Min => Min::op_fn,
            Self::Max => Max::op_fn,
        }
    }
    pub(super) const fn op_fn_inverse(&self) -> fn(f64, f64) -> f64 {
        match self {
            Self::Add => Add::op_fn_inverse,
            Self::Sub => Sub::op_fn_inverse,
            Self::Mul => Mul::op_fn_inverse,
            Self::Div => Div::op_fn_inverse,
            Self::Pow => Pow::op_fn_inverse,
            Self::Min => Min::op_fn_inverse,
            Self::Max => Max::op_fn_inverse,
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
