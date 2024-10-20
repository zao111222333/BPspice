use std::sync::{Arc, RwLock};

use ordered_float::OrderedFloat;

use super::{ChangeMarker, Expression, GradId, Tensor};

#[derive(Clone, Debug)]
pub enum Op {
    /// new assign
    Assgin,
    // Copy(Expression),
    Powf(Expression, f64),
    // (cond)? when_true : when_false
    Cond(Expression, Expression, Expression),
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
    fn fn_op(x: f64) -> f64;
}

struct Neg;
impl UnaryOpT for Neg {
    const OP: UnaryOp = UnaryOp::Neg;
    fn fn_op(x: f64) -> f64 {
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
    fn fn_op(x: f64) -> f64 {
        x.sin()
    }
}
struct Cos;
impl UnaryOpT for Cos {
    const OP: UnaryOp = UnaryOp::Cos;
    fn fn_op(x: f64) -> f64 {
        x.cos()
    }
}
struct Tanh;
impl UnaryOpT for Tanh {
    const OP: UnaryOp = UnaryOp::Tanh;
    fn fn_op(x: f64) -> f64 {
        x.tanh()
    }
}
struct Tan;
impl UnaryOpT for Tan {
    const OP: UnaryOp = UnaryOp::Tan;
    fn fn_op(x: f64) -> f64 {
        x.tan()
    }
}
struct Ceil;
impl UnaryOpT for Ceil {
    const OP: UnaryOp = UnaryOp::Ceil;
    fn fn_op(x: f64) -> f64 {
        x.ceil()
    }
}
struct Floor;
impl UnaryOpT for Floor {
    const OP: UnaryOp = UnaryOp::Floor;
    fn fn_op(x: f64) -> f64 {
        x.floor()
    }
}
struct Round;
impl UnaryOpT for Round {
    const OP: UnaryOp = UnaryOp::Round;
    fn fn_op(x: f64) -> f64 {
        x.round()
    }
}
struct Sign;
impl UnaryOpT for Sign {
    const OP: UnaryOp = UnaryOp::Sign;
    fn fn_op(x: f64) -> f64 {
        x.signum()
    }
}
struct Sqrt;
impl UnaryOpT for Sqrt {
    const OP: UnaryOp = UnaryOp::Sqrt;
    fn fn_op(x: f64) -> f64 {
        x.sqrt()
    }
}
struct Sqr;
impl UnaryOpT for Sqr {
    const OP: UnaryOp = UnaryOp::Sqr;
    fn fn_op(x: f64) -> f64 {
        x.powi(2)
    }
}
struct Log;
impl UnaryOpT for Log {
    const OP: UnaryOp = UnaryOp::Log;
    fn fn_op(x: f64) -> f64 {
        x.ln()
    }
}
struct Exp;
impl UnaryOpT for Exp {
    const OP: UnaryOp = UnaryOp::Exp;
    fn fn_op(x: f64) -> f64 {
        x.exp()
    }
}
struct Abs;
impl UnaryOpT for Abs {
    const OP: UnaryOp = UnaryOp::Abs;
    fn fn_op(x: f64) -> f64 {
        x.abs()
    }
}
struct Erf;
impl UnaryOpT for Erf {
    const OP: UnaryOp = UnaryOp::Erf;
    fn fn_op(x: f64) -> f64 {
        candle_core::cpu::erf::erf(x)
    }
}

impl UnaryOp {
    pub(super) const fn fn_op(&self) -> fn(f64) -> f64 {
        match self {
            Self::Neg => Neg::fn_op,
            Self::Sin => Sin::fn_op,
            Self::Cos => Cos::fn_op,
            Self::Tanh => Tanh::fn_op,
            Self::Tan => Tan::fn_op,
            Self::Ceil => Ceil::fn_op,
            Self::Floor => Floor::fn_op,
            Self::Round => Round::fn_op,
            Self::Sign => Sign::fn_op,
            Self::Sqrt => Sqrt::fn_op,
            Self::Sqr => Sqr::fn_op,
            Self::Log => Log::fn_op,
            Self::Exp => Exp::fn_op,
            Self::Abs => Abs::fn_op,
            Self::Erf => Erf::fn_op,
        }
    }
}

impl Tensor {
    pub(super) fn iter_unary_op(&self, fn_op: fn(f64) -> f64) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .map(|x| fn_op(*x))
            .collect()
    }
    pub(super) fn unary_op(&self, fn_op: fn(f64) -> f64, op: Op) -> Self {
        Self(Arc::new((
            if self.grad_id().is_some() {
                Some(GradId::new())
            } else {
                None
            },
            RwLock::new(self.iter_unary_op(fn_op)),
            ChangeMarker::new(),
            op,
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
            Expression::Const(x) => Expression::Const(T::fn_op(*x)),
            Expression::Tensor(tensor) => Expression::Tensor(
                tensor.unary_op(T::fn_op, Op::Unary(Self::Tensor(tensor.clone()), T::OP)),
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
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64;
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64;
    fn fn_backward_lhs(lhs: f64, rhs: f64, grad: f64, lhs_grad: &mut f64);
    fn fn_backward_rhs(lhs: f64, rhs: f64, grad: f64, rhs_grad: &mut f64);
}

struct Add;
impl BinaryOpT for Add {
    const OP: BinaryOp = BinaryOp::Add;
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs + rhs
    }
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs + rhs
    }
    fn fn_backward_lhs(_lhs: f64, _rhs: f64, grad: f64, lhs_grad: &mut f64) {
        *lhs_grad += grad;
    }
    fn fn_backward_rhs(_lhs: f64, _rhs: f64, grad: f64, rhs_grad: &mut f64) {
        *rhs_grad += grad;
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
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs - rhs
    }
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs - rhs
    }
    fn fn_backward_lhs(_lhs: f64, _rhs: f64, grad: f64, lhs_grad: &mut f64) {
        *lhs_grad += grad;
    }
    fn fn_backward_rhs(_lhs: f64, _rhs: f64, grad: f64, rhs_grad: &mut f64) {
        *rhs_grad -= grad;
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
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs * rhs
    }
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs * rhs
    }
    fn fn_backward_lhs(_lhs: f64, rhs: f64, grad: f64, lhs_grad: &mut f64) {
        *lhs_grad += grad * rhs;
    }
    fn fn_backward_rhs(lhs: f64, _rhs: f64, grad: f64, rhs_grad: &mut f64) {
        *rhs_grad += grad * lhs;
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
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs / rhs
    }
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs / rhs
    }
    fn fn_backward_lhs(_lhs: f64, rhs: f64, grad: f64, lhs_grad: &mut f64) {
        *lhs_grad += grad / rhs;
    }
    fn fn_backward_rhs(lhs: f64, rhs: f64, grad: f64, rhs_grad: &mut f64) {
        *rhs_grad -= grad * lhs / (rhs.powi(2));
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
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.powf(rhs)
    }
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.powf(rhs)
    }
    /// $ c = a^b $
    ///
    /// $\frac{\partial f}{\partial a} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial a} = \frac{\partial f}{\partial c} \cdot b \cdot a^{b - 1}$
    ///
    fn fn_backward_lhs(lhs: f64, rhs: f64, grad: f64, lhs_grad: &mut f64) {
        *lhs_grad += grad * rhs * lhs.powf(rhs - 1.0);
    }
    /// $\frac{\partial f}{\partial b} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial b} = \frac{\partial f}{\partial c} \cdot c \cdot \ln(a)$
    fn fn_backward_rhs(lhs: f64, rhs: f64, grad: f64, rhs_grad: &mut f64) {
        *rhs_grad += grad * lhs.powf(rhs) * f64::ln(lhs);
    }
}

struct Min;
impl BinaryOpT for Min {
    const OP: BinaryOp = BinaryOp::Min;
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.min(rhs)
    }
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.min(rhs)
    }
    fn fn_backward_lhs(lhs: f64, rhs: f64, grad: f64, lhs_grad: &mut f64) {
        match OrderedFloat(lhs).cmp(&OrderedFloat(rhs)) {
            std::cmp::Ordering::Less => *lhs_grad += grad,
            std::cmp::Ordering::Equal => *lhs_grad += grad / 2.0,
            std::cmp::Ordering::Greater => (),
        }
    }
    fn fn_backward_rhs(lhs: f64, rhs: f64, grad: f64, rhs_grad: &mut f64) {
        match OrderedFloat(rhs).cmp(&OrderedFloat(lhs)) {
            std::cmp::Ordering::Less => *rhs_grad += grad,
            std::cmp::Ordering::Equal => *rhs_grad += grad / 2.0,
            std::cmp::Ordering::Greater => (),
        }
    }
}
struct Max;
impl BinaryOpT for Max {
    const OP: BinaryOp = BinaryOp::Max;
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.max(rhs)
    }
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.max(rhs)
    }
    fn fn_backward_lhs(lhs: f64, rhs: f64, grad: f64, lhs_grad: &mut f64) {
        match OrderedFloat(lhs).cmp(&OrderedFloat(rhs)) {
            std::cmp::Ordering::Less => (),
            std::cmp::Ordering::Equal => *lhs_grad += grad / 2.0,
            std::cmp::Ordering::Greater => *lhs_grad += grad,
        }
    }
    fn fn_backward_rhs(lhs: f64, rhs: f64, grad: f64, rhs_grad: &mut f64) {
        match OrderedFloat(rhs).cmp(&OrderedFloat(lhs)) {
            std::cmp::Ordering::Less => (),
            std::cmp::Ordering::Equal => *rhs_grad += grad / 2.0,
            std::cmp::Ordering::Greater => *rhs_grad += grad,
        }
    }
}

impl BinaryOp {
    pub(super) const fn fn_lhs_op_rhs(&self) -> fn(f64, f64) -> f64 {
        match self {
            Self::Add => Add::fn_lhs_op_rhs,
            Self::Sub => Sub::fn_lhs_op_rhs,
            Self::Mul => Mul::fn_lhs_op_rhs,
            Self::Div => Div::fn_lhs_op_rhs,
            Self::Pow => Pow::fn_lhs_op_rhs,
            Self::Min => Min::fn_lhs_op_rhs,
            Self::Max => Max::fn_lhs_op_rhs,
        }
    }
    pub(super) const fn fn_rhs_op_lhs(&self) -> fn(f64, f64) -> f64 {
        match self {
            Self::Add => Add::fn_rhs_op_lhs,
            Self::Sub => Sub::fn_rhs_op_lhs,
            Self::Mul => Mul::fn_rhs_op_lhs,
            Self::Div => Div::fn_rhs_op_lhs,
            Self::Pow => Pow::fn_rhs_op_lhs,
            Self::Min => Min::fn_rhs_op_lhs,
            Self::Max => Max::fn_rhs_op_lhs,
        }
    }
    pub(super) const fn fn_backward(&self) -> [fn(f64, f64, f64, &mut f64); 2] {
        match self {
            Self::Add => [Add::fn_backward_lhs, Add::fn_backward_rhs],
            Self::Sub => [Sub::fn_backward_lhs, Sub::fn_backward_rhs],
            Self::Mul => [Mul::fn_backward_lhs, Mul::fn_backward_rhs],
            Self::Div => [Div::fn_backward_lhs, Div::fn_backward_rhs],
            Self::Pow => [Pow::fn_backward_lhs, Pow::fn_backward_rhs],
            Self::Min => [Min::fn_backward_lhs, Min::fn_backward_rhs],
            Self::Max => [Max::fn_backward_lhs, Max::fn_backward_rhs],
        }
    }
}

impl Tensor {
    pub(super) fn iter_binary_op(&self, rhs: &Self, fn_op: fn(f64, f64) -> f64) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .zip(rhs.values().read().unwrap().iter())
            .map(|(v1, v2)| fn_op(*v1, *v2))
            .collect()
    }
    pub(super) fn broadcast_iter_binary_op(
        &self,
        rhs: f64,
        fn_op: fn(f64, f64) -> f64,
    ) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .map(|v| fn_op(*v, rhs))
            .collect()
    }
    pub(super) fn binary_op(&self, rhs: &Self, fn_op: fn(f64, f64) -> f64, op: Op) -> Self {
        Self(Arc::new((
            if self.grad_id().is_some() || rhs.grad_id().is_some() {
                Some(GradId::new())
            } else {
                None
            },
            RwLock::new(self.iter_binary_op(rhs, fn_op)),
            ChangeMarker::new(),
            op,
        )))
    }
    pub(super) fn broadcast_binary_op(&self, rhs: f64, fn_op: fn(f64, f64) -> f64, op: Op) -> Self {
        Self(Arc::new((
            if self.grad_id().is_some() {
                Some(GradId::new())
            } else {
                None
            },
            RwLock::new(self.broadcast_iter_binary_op(rhs, fn_op)),
            ChangeMarker::new(),
            op,
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
            (Self::Const(v1), Self::Const(v2)) => Self::Const(T::fn_lhs_op_rhs(*v1, *v2)),
            (Self::Const(v1), Self::Tensor(tensor2)) => Self::Tensor(tensor2.broadcast_binary_op(
                *v1,
                T::fn_rhs_op_lhs,
                Op::Binary(Self::Const(*v1), Self::Tensor(tensor2.clone()), T::OP),
            )),
            (Self::Tensor(tensor1), Self::Const(v2)) => Self::Tensor(tensor1.broadcast_binary_op(
                *v2,
                T::fn_lhs_op_rhs,
                Op::Binary(Self::Tensor(tensor1.clone()), Self::Const(*v2), T::OP),
            )),
            (Expression::Tensor(tensor1), Expression::Tensor(tensor2)) => {
                Self::Tensor(tensor1.binary_op(
                    tensor2,
                    T::fn_lhs_op_rhs,
                    Op::Binary(
                        Self::Tensor(tensor1.clone()),
                        Self::Tensor(tensor2.clone()),
                        T::OP,
                    ),
                ))
            }
        }
    }
}
