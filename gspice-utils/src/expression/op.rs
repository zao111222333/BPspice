use std::{
    cmp::Ordering,
    sync::{Arc, RwLock},
};

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
    // Cmp(Expression, CmpOp),
    Unary(Expression, UnaryOp),
    Binary(Expression, Expression, BinaryOp),
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   Powf   ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

pub(super) struct Powf;
impl Powf {
    pub(super) fn fn_op(x: f64, n: f64) -> f64 {
        x.powf(n)
    }
    pub(super) fn fn_backward(x: &f64, n: f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * n * x.powf(n - 1.0);
    }
}
impl Expression {
    #[inline]
    pub fn powf(&self, n: f64) -> Self {
        match self {
            Self::Const(x) => Self::Const(Powf::fn_op(*x, n)),
            Self::Tensor(tensor) => Self::Tensor(tensor.broadcast_binary_op(
                n,
                Powf::fn_op,
                Op::Powf(Self::Tensor(tensor.clone()), n),
            )),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   UnaryOp   ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Clone, Copy, Debug)]
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
    Cubic,
    Log,
    Exp,
    Abs,
    Erf,
}

trait UnaryOpT {
    const OP: UnaryOp;
    fn fn_op(x: f64) -> f64;
    fn fn_backward(x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64);
}

struct Neg;
impl UnaryOpT for Neg {
    const OP: UnaryOp = UnaryOp::Neg;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        -x
    }
    #[inline]
    fn fn_backward(_x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad -= grad;
    }
}
impl<'a> core::ops::Neg for &'a Expression {
    type Output = Expression;

    #[inline]
    fn neg(self) -> Self::Output {
        Expression::unary_op::<Neg>(&self)
    }
}

struct Sin;
impl UnaryOpT for Sin {
    const OP: UnaryOp = UnaryOp::Sin;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.sin()
    }
    #[inline]
    fn fn_backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * x.cos();
    }
}
struct Cos;
impl UnaryOpT for Cos {
    const OP: UnaryOp = UnaryOp::Cos;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.cos()
    }
    #[inline]
    fn fn_backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad -= grad * x.sin();
    }
}
struct Tanh;
impl UnaryOpT for Tanh {
    const OP: UnaryOp = UnaryOp::Tanh;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.tanh()
    }
    // $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial x} = \frac{\partial f}{\partial c} \cdot (1 - \tanh^2(x))$
    #[inline]
    fn fn_backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        let minus_dtanh = res * res - 1.;
        *sum_grad -= grad * minus_dtanh;
    }
}
struct Tan;
impl UnaryOpT for Tan {
    const OP: UnaryOp = UnaryOp::Tan;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.tan()
    }
    /// $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial x} = \frac{\partial f}{\partial c} \cdot (1 + \tan^2(x))$
    #[inline]
    fn fn_backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        let dtan = res * res + 1.;
        *sum_grad -= grad * dtan;
    }
}
struct Ceil;
impl UnaryOpT for Ceil {
    const OP: UnaryOp = UnaryOp::Ceil;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.ceil()
    }
    // FIXME: No gradient for compare
    #[inline]
    fn fn_backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
        log::error!("BackwardNotSupported Ceil");
        // *sum_grad += grad;
    }
}
struct Floor;
impl UnaryOpT for Floor {
    const OP: UnaryOp = UnaryOp::Floor;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.floor()
    }
    #[inline]
    fn fn_backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
        log::error!("BackwardNotSupported Floor");
        // *sum_grad += grad;
    }
}
struct Round;
impl UnaryOpT for Round {
    const OP: UnaryOp = UnaryOp::Round;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.round()
    }
    #[inline]
    fn fn_backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
        log::error!("BackwardNotSupported Round");
        // *sum_grad += grad;
    }
}
struct Sign;
impl UnaryOpT for Sign {
    const OP: UnaryOp = UnaryOp::Sign;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.signum()
    }
    #[inline]
    fn fn_backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
        log::error!("BackwardNotSupported Sign");
        // let epsilon = 1e-10;
        // if (x.abs() - epsilon).is_sign_negative() {
        //     *sum_grad += grad;
        // }
    }
}
struct Sqrt;
impl UnaryOpT for Sqrt {
    const OP: UnaryOp = UnaryOp::Sqrt;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.sqrt()
    }
    #[inline]
    fn fn_backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * 0.5 / res;
    }
}
struct Sqr;
impl UnaryOpT for Sqr {
    const OP: UnaryOp = UnaryOp::Sqr;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x * x
    }
    #[inline]
    fn fn_backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * 2.0 * x;
    }
}
struct Cubic;
impl UnaryOpT for Cubic {
    const OP: UnaryOp = UnaryOp::Cubic;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x * x * x
    }
    #[inline]
    fn fn_backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * 3.0 * x * x;
    }
}

struct Log;
impl UnaryOpT for Log {
    const OP: UnaryOp = UnaryOp::Log;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.ln()
    }
    #[inline]
    fn fn_backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad / x;
    }
}
struct Exp;
impl UnaryOpT for Exp {
    const OP: UnaryOp = UnaryOp::Exp;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.exp()
    }
    #[inline]
    fn fn_backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * res;
    }
}
struct Abs;
impl UnaryOpT for Abs {
    const OP: UnaryOp = UnaryOp::Abs;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        x.abs()
    }
    #[inline]
    fn fn_backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        if x.is_sign_positive() {
            *sum_grad += grad;
        } else {
            *sum_grad -= grad;
        }
    }
}
struct Erf;
impl UnaryOpT for Erf {
    const OP: UnaryOp = UnaryOp::Erf;
    #[inline]
    fn fn_op(x: f64) -> f64 {
        candle_core::cpu::erf::erf(x)
    }
    #[inline]
    fn fn_backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
        let erf_grad = (2. / std::f64::consts::PI.sqrt()) * (-x * x).exp();
        *sum_grad += grad * erf_grad;
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
            Self::Cubic => Cubic::fn_op,
            Self::Log => Log::fn_op,
            Self::Exp => Exp::fn_op,
            Self::Abs => Abs::fn_op,
            Self::Erf => Erf::fn_op,
        }
    }
    #[inline]
    pub(super) const fn fn_backward(&self) -> fn(&f64, &f64, &f64, &mut f64) {
        match self {
            Self::Neg => Neg::fn_backward,
            Self::Sin => Sin::fn_backward,
            Self::Cos => Cos::fn_backward,
            Self::Tanh => Tanh::fn_backward,
            Self::Tan => Tan::fn_backward,
            Self::Ceil => Ceil::fn_backward,
            Self::Floor => Floor::fn_backward,
            Self::Round => Round::fn_backward,
            Self::Sign => Sign::fn_backward,
            Self::Sqrt => Sqrt::fn_backward,
            Self::Sqr => Sqr::fn_backward,
            Self::Cubic => Cubic::fn_backward,
            Self::Log => Log::fn_backward,
            Self::Exp => Exp::fn_backward,
            Self::Abs => Abs::fn_backward,
            Self::Erf => Erf::fn_backward,
        }
    }
}

impl Tensor {
    #[inline]
    pub(super) fn iter_unary_op(&self, fn_op: fn(f64) -> f64) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .map(|x| fn_op(*x))
            .collect()
    }
    #[inline]
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
    #[inline]
    pub fn sin(&self) -> Self {
        Expression::unary_op::<Sin>(&self)
    }
    #[inline]
    pub fn cos(&self) -> Self {
        Expression::unary_op::<Cos>(&self)
    }
    #[inline]
    pub fn tanh(&self) -> Self {
        Expression::unary_op::<Tanh>(&self)
    }
    #[inline]
    pub fn tan(&self) -> Self {
        Expression::unary_op::<Tan>(&self)
    }
    #[inline]
    pub fn ceil(&self) -> Self {
        Expression::unary_op::<Ceil>(&self)
    }
    #[inline]
    pub fn floor(&self) -> Self {
        Expression::unary_op::<Floor>(&self)
    }
    #[inline]
    pub fn round(&self) -> Self {
        Expression::unary_op::<Round>(&self)
    }
    #[inline]
    pub fn sign(&self) -> Self {
        Expression::unary_op::<Sign>(&self)
    }
    #[inline]
    pub fn sqrt(&self) -> Self {
        Expression::unary_op::<Sqrt>(&self)
    }
    #[inline]
    pub fn sqr(&self) -> Self {
        Expression::unary_op::<Sqr>(&self)
    }
    #[inline]
    pub fn cubic(&self) -> Self {
        Expression::unary_op::<Cubic>(&self)
    }
    #[inline]
    pub fn log(&self) -> Self {
        Expression::unary_op::<Log>(&self)
    }
    #[inline]
    pub fn exp(&self) -> Self {
        Expression::unary_op::<Exp>(&self)
    }
    #[inline]
    pub fn abs(&self) -> Self {
        Expression::unary_op::<Abs>(&self)
    }
    #[inline]
    pub fn erf(&self) -> Self {
        Expression::unary_op::<Erf>(&self)
    }
    #[inline]
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
    Cmp(CmpOp),
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
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64;
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64;
    fn fn_backward_lhs(lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64);
    fn fn_backward_rhs(lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64);
}

struct Eq;
impl BinaryOpT for Eq {
    const OP: BinaryOp = BinaryOp::Cmp(CmpOp::Eq);
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).eq(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).eq(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    // FIXME: No gradient for compare
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _lhs_sum_grad: &mut f64) {}
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _rhs_sum_grad: &mut f64) {}
}
struct Ne;
impl BinaryOpT for Ne {
    const OP: BinaryOp = BinaryOp::Cmp(CmpOp::Ne);
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).ne(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).ne(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    // FIXME: No gradient for compare
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _lhs_sum_grad: &mut f64) {}
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _rhs_sum_grad: &mut f64) {}
}
struct Le;
impl BinaryOpT for Le {
    const OP: BinaryOp = BinaryOp::Cmp(CmpOp::Le);
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).le(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).le(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    // FIXME: No gradient for compare
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _lhs_sum_grad: &mut f64) {}
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _rhs_sum_grad: &mut f64) {}
}
struct Ge;
impl BinaryOpT for Ge {
    const OP: BinaryOp = BinaryOp::Cmp(CmpOp::Ge);
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).ge(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).ge(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    // FIXME: No gradient for compare
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _lhs_sum_grad: &mut f64) {}
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _rhs_sum_grad: &mut f64) {}
}
struct Lt;
impl BinaryOpT for Lt {
    const OP: BinaryOp = BinaryOp::Cmp(CmpOp::Lt);
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).lt(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).lt(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    // FIXME: No gradient for compare
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _lhs_sum_grad: &mut f64) {}
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _rhs_sum_grad: &mut f64) {}
}
struct Gt;
impl BinaryOpT for Gt {
    const OP: BinaryOp = BinaryOp::Cmp(CmpOp::Gt);
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).gt(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        use num_traits::identities::{One, Zero};
        if OrderedFloat(lhs).gt(&OrderedFloat(rhs)) {
            f64::one()
        } else {
            f64::zero()
        }
    }
    // FIXME: No gradient for compare
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _lhs_sum_grad: &mut f64) {}
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, _grad: &f64, _rhs_sum_grad: &mut f64) {}
}

struct Add;
impl BinaryOpT for Add {
    const OP: BinaryOp = BinaryOp::Add;
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs + rhs
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs + rhs
    }
    #[inline]
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad;
    }
    #[inline]
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad;
    }
}
impl<'a, 'b> core::ops::Add<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn add(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Add>(rhs)
    }
}

struct Sub;
impl BinaryOpT for Sub {
    const OP: BinaryOp = BinaryOp::Sub;
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs - rhs
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs - rhs
    }
    #[inline]
    fn fn_backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad;
    }
    #[inline]
    fn fn_backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad -= grad;
    }
}
impl<'a, 'b> core::ops::Sub<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn sub(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Sub>(rhs)
    }
}

struct Mul;
impl BinaryOpT for Mul {
    const OP: BinaryOp = BinaryOp::Mul;
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs * rhs
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs * rhs
    }
    #[inline]
    fn fn_backward_lhs(_lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad * rhs;
    }
    #[inline]
    fn fn_backward_rhs(lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad * lhs;
    }
}
impl<'a, 'b> core::ops::Mul<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn mul(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Mul>(rhs)
    }
}

struct Div;
impl BinaryOpT for Div {
    const OP: BinaryOp = BinaryOp::Div;
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs / rhs
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs / rhs
    }
    #[inline]
    fn fn_backward_lhs(_lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad / rhs;
    }
    #[inline]
    fn fn_backward_rhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad -= grad * lhs / (rhs * rhs);
    }
}
impl<'a, 'b> core::ops::Div<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn div(self, rhs: &'b Expression) -> Expression {
        self.binary_op::<Div>(rhs)
    }
}

struct Pow;
impl BinaryOpT for Pow {
    const OP: BinaryOp = BinaryOp::Pow;
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.powf(rhs)
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.powf(rhs)
    }
    /// $ c = a^b $
    ///
    /// $\frac{\partial f}{\partial a} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial a} = \frac{\partial f}{\partial c} \cdot b \cdot a^{b - 1}$
    ///
    #[inline]
    fn fn_backward_lhs(lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad * rhs * res / lhs;
        // *lhs_sum_grad += grad * rhs * lhs.powf(rhs - 1.0);
    }
    /// $\frac{\partial f}{\partial b} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial b} = \frac{\partial f}{\partial c} \cdot c \cdot \ln(a)$
    #[inline]
    fn fn_backward_rhs(lhs: &f64, _rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad * res * (lhs.ln());
    }
}

struct Min;
impl BinaryOpT for Min {
    const OP: BinaryOp = BinaryOp::Min;
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.min(rhs)
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.min(rhs)
    }
    #[inline]
    fn fn_backward_lhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        // If both masks are 1 one the same point, we want to scale the
        // gradient by 0.5 rather than 1.
        match OrderedFloat(*lhs).cmp(&OrderedFloat(*rhs)) {
            Ordering::Less => *lhs_sum_grad += grad,
            Ordering::Equal => *lhs_sum_grad += grad / 2.0,
            Ordering::Greater => (),
        }
    }
    #[inline]
    fn fn_backward_rhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        // If both masks are 1 one the same point, we want to scale the
        // gradient by 0.5 rather than 1.
        match OrderedFloat(*rhs).cmp(&OrderedFloat(*lhs)) {
            Ordering::Less => *rhs_sum_grad += grad,
            Ordering::Equal => *rhs_sum_grad += grad / 2.0,
            Ordering::Greater => (),
        }
    }
}
struct Max;
impl BinaryOpT for Max {
    const OP: BinaryOp = BinaryOp::Max;
    #[inline]
    fn fn_lhs_op_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.max(rhs)
    }
    #[inline]
    fn fn_rhs_op_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.max(rhs)
    }
    #[inline]
    fn fn_backward_lhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        // If both masks are 1 one the same point, we want to scale the
        // gradient by 0.5 rather than 1.
        match OrderedFloat(*lhs).cmp(&OrderedFloat(*rhs)) {
            Ordering::Less => (),
            Ordering::Equal => *lhs_sum_grad += grad / 2.0,
            Ordering::Greater => *lhs_sum_grad += grad,
        }
    }
    #[inline]
    fn fn_backward_rhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        // If both masks are 1 one the same point, we want to scale the
        // gradient by 0.5 rather than 1.
        match OrderedFloat(*rhs).cmp(&OrderedFloat(*lhs)) {
            Ordering::Less => (),
            Ordering::Equal => *rhs_sum_grad += grad / 2.0,
            Ordering::Greater => *rhs_sum_grad += grad,
        }
    }
}

impl BinaryOp {
    #[inline]
    pub(super) const fn fn_lhs_op_rhs(&self) -> fn(f64, f64) -> f64 {
        match self {
            Self::Add => Add::fn_lhs_op_rhs,
            Self::Sub => Sub::fn_lhs_op_rhs,
            Self::Mul => Mul::fn_lhs_op_rhs,
            Self::Div => Div::fn_lhs_op_rhs,
            Self::Pow => Pow::fn_lhs_op_rhs,
            Self::Min => Min::fn_lhs_op_rhs,
            Self::Max => Max::fn_lhs_op_rhs,
            Self::Cmp(cmp_op) => match cmp_op {
                CmpOp::Eq => Eq::fn_lhs_op_rhs,
                CmpOp::Ne => Ne::fn_lhs_op_rhs,
                CmpOp::Le => Le::fn_lhs_op_rhs,
                CmpOp::Ge => Ge::fn_lhs_op_rhs,
                CmpOp::Lt => Lt::fn_lhs_op_rhs,
                CmpOp::Gt => Gt::fn_lhs_op_rhs,
            },
        }
    }
    #[inline]
    pub(super) const fn fn_rhs_op_lhs(&self) -> fn(f64, f64) -> f64 {
        match self {
            Self::Add => Add::fn_rhs_op_lhs,
            Self::Sub => Sub::fn_rhs_op_lhs,
            Self::Mul => Mul::fn_rhs_op_lhs,
            Self::Div => Div::fn_rhs_op_lhs,
            Self::Pow => Pow::fn_rhs_op_lhs,
            Self::Min => Min::fn_rhs_op_lhs,
            Self::Max => Max::fn_rhs_op_lhs,
            Self::Cmp(cmp_op) => match cmp_op {
                CmpOp::Eq => Eq::fn_rhs_op_lhs,
                CmpOp::Ne => Ne::fn_rhs_op_lhs,
                CmpOp::Le => Le::fn_rhs_op_lhs,
                CmpOp::Ge => Ge::fn_rhs_op_lhs,
                CmpOp::Lt => Lt::fn_rhs_op_lhs,
                CmpOp::Gt => Gt::fn_rhs_op_lhs,
            },
        }
    }
    #[inline]
    pub(super) const fn fn_backward(&self) -> [fn(&f64, &f64, &f64, &f64, &mut f64); 2] {
        match self {
            Self::Add => [Add::fn_backward_lhs, Add::fn_backward_rhs],
            Self::Sub => [Sub::fn_backward_lhs, Sub::fn_backward_rhs],
            Self::Mul => [Mul::fn_backward_lhs, Mul::fn_backward_rhs],
            Self::Div => [Div::fn_backward_lhs, Div::fn_backward_rhs],
            Self::Pow => [Pow::fn_backward_lhs, Pow::fn_backward_rhs],
            Self::Min => [Min::fn_backward_lhs, Min::fn_backward_rhs],
            Self::Max => [Max::fn_backward_lhs, Max::fn_backward_rhs],
            Self::Cmp(cmp_op) => match cmp_op {
                CmpOp::Eq => [Eq::fn_backward_lhs, Eq::fn_backward_rhs],
                CmpOp::Ne => [Ne::fn_backward_lhs, Ne::fn_backward_rhs],
                CmpOp::Le => [Le::fn_backward_lhs, Le::fn_backward_rhs],
                CmpOp::Ge => [Ge::fn_backward_lhs, Ge::fn_backward_rhs],
                CmpOp::Lt => [Lt::fn_backward_lhs, Lt::fn_backward_rhs],
                CmpOp::Gt => [Gt::fn_backward_lhs, Gt::fn_backward_rhs],
            },
        }
    }
}

impl Tensor {
    #[inline]
    pub(super) fn iter_binary_op(&self, rhs: &Self, fn_op: fn(f64, f64) -> f64) -> Vec<f64> {
        let self_vec = self.values().read().unwrap();
        let rhs_vec = rhs.values().read().unwrap();
        assert_eq!(rhs_vec.len(), self_vec.len(), "tensor length mismatch!");
        self_vec
            .iter()
            .zip(rhs_vec.iter())
            .map(|(v1, v2)| fn_op(*v1, *v2))
            .collect()
    }
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn pow(&self, rhs: &Self) -> Self {
        self.binary_op::<Pow>(rhs)
    }
    #[inline]
    pub fn min(&self, rhs: &Self) -> Self {
        self.binary_op::<Min>(rhs)
    }
    #[inline]
    pub fn max(&self, rhs: &Self) -> Self {
        self.binary_op::<Max>(rhs)
    }
    #[inline]
    pub fn eq(&self, rhs: &Self) -> Self {
        self.binary_op::<Eq>(rhs)
    }
    #[inline]
    pub fn ne(&self, rhs: &Self) -> Self {
        self.binary_op::<Ne>(rhs)
    }
    #[inline]
    pub fn le(&self, rhs: &Self) -> Self {
        self.binary_op::<Le>(rhs)
    }
    #[inline]
    pub fn ge(&self, rhs: &Self) -> Self {
        self.binary_op::<Ge>(rhs)
    }
    #[inline]
    pub fn lt(&self, rhs: &Self) -> Self {
        self.binary_op::<Lt>(rhs)
    }
    #[inline]
    pub fn gt(&self, rhs: &Self) -> Self {
        self.binary_op::<Gt>(rhs)
    }
    #[inline]
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
