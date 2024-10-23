use itertools::izip;
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;
use std::{cmp::Ordering, fmt::Debug};

use super::{Expression, GradId, Tensor};

#[derive(Debug)]
pub enum Op {
    /// new assign
    Assgin,
    Powf(Expression, f64),
    /// `(cond)? on_true : on_false`
    ///
    /// smoothing method:
    /// `cond*on_true + (1-cond)*on_false`
    Cond(Expression, Expression, Expression),
    Unary(Expression, UnaryOp),
    Binary(Expression, Expression, BinaryOp),
    DiscreteBinary(Expression, Expression, DiscreteBinaryOp, GradMethod),
    // DiscreteUnary(Expression, DiscreteUnaryOp, GradMethod),
}

/// GradMethod only activate in gradient mode
#[derive(Clone, Copy, Debug)]
pub enum GradMethod {
    Discrete,
    Linear(GradMethodLinear),
    Sigmoid(GradMethodSigmoid),
}

impl GradMethod {
    #[inline]
    fn new_sigmoid(k: f64) -> Self {
        assert!(k.is_sign_positive());
        Self::Sigmoid(GradMethodSigmoid { k })
    }
    #[inline]
    fn new_linear(epsilon: f64) -> Self {
        assert!(epsilon.is_sign_positive());
        Self::Linear(GradMethodLinear { epsilon })
    }
}

macro_rules! assert_logic {
    ($logic:expr) => {
        debug_assert!(OrderedFloat($logic).ge(&OrderedFloat(0.0)));
        debug_assert!(OrderedFloat($logic).le(&OrderedFloat(1.0)));
    };
}

macro_rules! assert_logic_tensor {
    ($tensor:expr) => {
        #[cfg(debug_assertions)]
        {
            assert!($tensor.is_logic(), "ASSERT logic (only in debug mode), if you ensure that is a logic tensor, use `mark_logic` to Expression");
        }
    };
}
macro_rules! mark_logic_tensor {
    ($tensor:expr) => {
        #[cfg(debug_assertions)]
        {
            $tensor.mark_logic();
            $tensor
        }
    };
}
////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   Powf   ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

pub(super) struct Powf;
impl Powf {
    pub(super) fn forward(x: f64, n: f64) -> f64 {
        x.powf(n)
    }
    pub(super) fn backward(x: &f64, n: f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * n * x.powf(n - 1.0);
    }
}
impl Expression {
    #[inline]
    pub fn powf(&self, n: f64) -> Self {
        match self {
            Self::Const(x) => Self::Const(Powf::forward(*x, n)),
            Self::Tensor(tensor) => Self::Tensor(tensor.broadcast_binary_op(
                n,
                Powf::forward,
                Op::Powf(Self::Tensor(tensor.clone()), n),
            )),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   Cond   ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

pub(super) struct Cond;
impl Cond {
    /// `(cond)? on_true : on_false`
    ///
    /// smoothing method:
    /// `cond*on_true + (1-cond)*on_false`
    #[inline]
    pub(super) fn forward(cond: &f64, on_true: f64, on_false: f64) -> f64 {
        assert_logic!(*cond);
        cond * on_true + (1.0 - cond) * on_false
    }
    /// $\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial a} = \text{grad\_output} \times (b - c)$
    #[inline]
    pub(super) fn backward_cond(
        _cond: &f64,
        on_true: &f64,
        on_false: &f64,
        grad: &f64,
        cond_grad: &mut f64,
    ) {
        *cond_grad += grad * (on_true - on_false);
    }
    #[inline]
    pub(super) fn backward_on_true(
        cond: &f64,
        _on_true: &f64,
        _on_false: &f64,
        grad: &f64,
        on_true_grad: &mut f64,
    ) {
        *on_true_grad += cond * grad;
    }
    #[inline]
    pub(super) fn backward_on_false(
        cond: &f64,
        _on_true: &f64,
        _on_false: &f64,
        grad: &f64,
        on_false_grad: &mut f64,
    ) {
        *on_false_grad += (1.0 - cond) * grad;
    }
    #[inline]
    pub(super) fn iter_tensor_x_x(
        cond_tensor: &Tensor,
        on_true_x: f64,
        on_false_x: f64,
    ) -> Vec<f64> {
        cond_tensor
            .values()
            .read()
            .unwrap()
            .iter()
            .map(|cond_x| Cond::forward(cond_x, on_true_x, on_false_x))
            .collect()
    }
    #[inline]
    pub(super) fn iter_tensor_x_tensor(
        cond_tensor: &Tensor,
        on_true_x: f64,
        on_false_tensor: &Tensor,
    ) -> Vec<f64> {
        izip!(
            cond_tensor.values().read().unwrap().iter(),
            on_false_tensor.values().read().unwrap().iter()
        )
        .map(|(cond_x, on_false_x)| Cond::forward(cond_x, on_true_x, *on_false_x))
        .collect()
    }
    #[inline]
    pub(super) fn iter_tensor_tensor_x(
        cond_tensor: &Tensor,
        on_true_tensor: &Tensor,
        on_false_x: f64,
    ) -> Vec<f64> {
        izip!(
            cond_tensor.values().read().unwrap().iter(),
            on_true_tensor.values().read().unwrap().iter(),
        )
        .map(|(cond_x, on_true_x)| Cond::forward(cond_x, *on_true_x, on_false_x))
        .collect()
    }
    #[inline]
    pub(super) fn iter_tensor_tensor_tensor(
        cond_tensor: &Tensor,
        on_true_tensor: &Tensor,
        on_false_tensor: &Tensor,
    ) -> Vec<f64> {
        izip!(
            cond_tensor.values().read().unwrap().iter(),
            on_true_tensor.values().read().unwrap().iter(),
            on_false_tensor.values().read().unwrap().iter()
        )
        .map(|(cond_x, on_true_x, on_false_x)| Cond::forward(cond_x, *on_true_x, *on_false_x))
        .collect()
    }
}

impl Expression {
    /// smoothing method
    /// `cond*on_true + (1-cond)*on_false`
    #[inline]
    pub fn cond(&self, on_true: &Self, on_false: &Self) -> Self {
        #[cfg(debug_assertions)]
        if let Self::Tensor(cond_tensor) = self {
            assert_logic_tensor!(cond_tensor);
        }
        match (self, on_true, on_false) {
            (Self::Const(cond_x), Self::Const(on_true_x), Self::Const(on_false_x)) => {
                Self::Const(Cond::forward(cond_x, *on_true_x, *on_false_x))
            }
            (Self::Const(cond_x), Self::Const(on_true_x), Self::Tensor(on_false_tensor)) => {
                if cond_x.is_zero() {
                    Self::Tensor(on_false_tensor.clone())
                } else {
                    Self::Const(*on_true_x)
                }
            }
            (Self::Const(cond_x), Self::Tensor(on_true_tensor), Self::Const(on_false_x)) => {
                if cond_x.is_zero() {
                    Self::Const(*on_false_x)
                } else {
                    Self::Tensor(on_true_tensor.clone())
                }
            }
            (Self::Const(cond_x), Self::Tensor(on_true_tensor), Self::Tensor(on_false_tensor)) => {
                if cond_x.is_zero() {
                    Self::Tensor(on_false_tensor.clone())
                } else {
                    Self::Tensor(on_true_tensor.clone())
                }
            }
            (Self::Tensor(cond_tensor), Self::Const(on_true_x), Self::Const(on_false_x)) => {
                Self::Tensor(Tensor::new(
                    if cond_tensor.with_grad() {
                        Some(GradId::new())
                    } else {
                        None
                    },
                    Cond::iter_tensor_x_x(cond_tensor, *on_true_x, *on_false_x),
                    Op::Cond(
                        Self::Tensor(cond_tensor.clone()),
                        Self::Const(*on_true_x),
                        Self::Const(*on_false_x),
                    ),
                ))
            }
            (Self::Tensor(cond_tensor), Self::Const(on_true_x), Self::Tensor(on_false_tensor)) => {
                Self::Tensor(Tensor::new(
                    if cond_tensor.with_grad() || on_false_tensor.with_grad() {
                        Some(GradId::new())
                    } else {
                        None
                    },
                    Cond::iter_tensor_x_tensor(cond_tensor, *on_true_x, on_false_tensor),
                    Op::Cond(
                        Self::Tensor(cond_tensor.clone()),
                        Self::Const(*on_true_x),
                        Self::Tensor(on_false_tensor.clone()),
                    ),
                ))
            }
            (Self::Tensor(cond_tensor), Self::Tensor(on_true_tensor), Self::Const(on_false_x)) => {
                Self::Tensor(Tensor::new(
                    if cond_tensor.with_grad() || on_true_tensor.with_grad() {
                        Some(GradId::new())
                    } else {
                        None
                    },
                    Cond::iter_tensor_tensor_x(cond_tensor, on_true_tensor, *on_false_x),
                    Op::Cond(
                        Self::Tensor(cond_tensor.clone()),
                        Self::Tensor(on_true_tensor.clone()),
                        Self::Const(*on_false_x),
                    ),
                ))
            }
            (
                Self::Tensor(cond_tensor),
                Self::Tensor(on_true_tensor),
                Self::Tensor(on_false_tensor),
            ) => Self::Tensor(Tensor::new(
                if cond_tensor.with_grad()
                    || on_true_tensor.with_grad()
                    || on_false_tensor.with_grad()
                {
                    Some(GradId::new())
                } else {
                    None
                },
                Cond::iter_tensor_tensor_tensor(cond_tensor, on_true_tensor, on_false_tensor),
                Op::Cond(
                    Self::Tensor(cond_tensor.clone()),
                    Self::Tensor(on_true_tensor.clone()),
                    Self::Tensor(on_false_tensor.clone()),
                ),
            )),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   UnaryOp   ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Clone, Copy, Debug)]
pub enum DiscreteUnaryOp {
    /// ``` text
    ///            __/
    ///         __/
    ///      __/
    /// ____/          0
    /// --------------->
    ///                x
    /// ```
    Ceil,
    /// ``` text
    ///            __/
    ///         __/
    ///      __/
    /// ____/          0
    /// --------------->
    ///                x
    /// ```
    Floor,
    /// ``` text
    ///            __/
    ///         __/
    ///      __/
    /// ____/          0
    /// --------------->
    ///                x
    /// ```
    Round,
    /// ``` text
    ///         _____  1
    ///        /
    ///       /        0
    ///      /
    /// ____/         -1
    /// --------------->
    ///        0       x
    /// ```
    Sign,
    /// penalty
    /// ``` text
    ///           /    slop = factor
    ///         /
    ///       /
    /// ____/          0
    /// --------------->
    ///   th           x
    /// ```
    Lt(Constraint),
    /// penalty
    /// ``` text
    ///   \            slop = -factor
    ///     \          
    ///       \
    ///         \___   0
    /// --------------->
    ///          th    x
    /// ```
    Gt(Constraint),
}
#[derive(Clone, Copy, Debug)]
pub struct Constraint {
    threshold: f64,
    factor: f64,
}
#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    LogicNot,
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
    #[inline]
    fn debug_assertions(tensor: &Tensor) {
        _ = tensor;
    }
    #[inline]
    fn debug_mark(tensor: Tensor) -> Tensor {
        tensor
    }
    fn forward(x: f64) -> f64;
    fn backward(x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64);
}

struct Neg;
impl UnaryOpT for Neg {
    const OP: UnaryOp = UnaryOp::Neg;
    #[inline]
    fn forward(x: f64) -> f64 {
        -x
    }
    #[inline]
    fn backward(_x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad -= grad;
    }
}

impl<'a> core::ops::Neg for &'a Expression {
    type Output = Expression;
    #[inline]
    fn neg(self) -> Self::Output {
        self.neg()
    }
}

struct Sin;
impl UnaryOpT for Sin {
    const OP: UnaryOp = UnaryOp::Sin;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.sin()
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * x.cos();
    }
}
struct Cos;
impl UnaryOpT for Cos {
    const OP: UnaryOp = UnaryOp::Cos;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.cos()
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad -= grad * x.sin();
    }
}
struct Tanh;
impl UnaryOpT for Tanh {
    const OP: UnaryOp = UnaryOp::Tanh;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.tanh()
    }
    // $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial x} = \frac{\partial f}{\partial c} \cdot (1 - \tanh^2(x))$
    #[inline]
    fn backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        let minus_dtanh = res * res - 1.;
        *sum_grad -= grad * minus_dtanh;
    }
}
struct Tan;
impl UnaryOpT for Tan {
    const OP: UnaryOp = UnaryOp::Tan;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.tan()
    }
    /// $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial x} = \frac{\partial f}{\partial c} \cdot (1 + \tan^2(x))$
    #[inline]
    fn backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        let dtan = res * res + 1.;
        *sum_grad -= grad * dtan;
    }
}
struct Ceil;
impl UnaryOpT for Ceil {
    const OP: UnaryOp = UnaryOp::Ceil;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.ceil()
    }
    // FIXME: No gradient for compare
    #[inline]
    fn backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
        log::error!("BackwardNotSupported Ceil");
        // *sum_grad += grad;
    }
}
struct Floor;
impl UnaryOpT for Floor {
    const OP: UnaryOp = UnaryOp::Floor;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.floor()
    }
    #[inline]
    fn backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
        log::error!("BackwardNotSupported Floor");
        // *sum_grad += grad;
    }
}

struct Round;
impl UnaryOpT for Round {
    const OP: UnaryOp = UnaryOp::Round;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.round()
    }
    #[inline]
    fn backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
        log::error!("BackwardNotSupported Round");
        // *sum_grad += grad;
    }
}
struct Sign;
impl UnaryOpT for Sign {
    const OP: UnaryOp = UnaryOp::Sign;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.signum()
    }
    #[inline]
    fn backward(_x: &f64, _res: &f64, _grad: &f64, _sum_grad: &mut f64) {
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
    fn forward(x: f64) -> f64 {
        x.sqrt()
    }
    #[inline]
    fn backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * 0.5 / res;
    }
}
struct Sqr;
impl UnaryOpT for Sqr {
    const OP: UnaryOp = UnaryOp::Sqr;
    #[inline]
    fn forward(x: f64) -> f64 {
        x * x
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * 2.0 * x;
    }
}
struct Cubic;
impl UnaryOpT for Cubic {
    const OP: UnaryOp = UnaryOp::Cubic;
    #[inline]
    fn forward(x: f64) -> f64 {
        x * x * x
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * 3.0 * x * x;
    }
}

struct Log;
impl UnaryOpT for Log {
    const OP: UnaryOp = UnaryOp::Log;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.ln()
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad / x;
    }
}
struct Exp;
impl UnaryOpT for Exp {
    const OP: UnaryOp = UnaryOp::Exp;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.exp()
    }
    #[inline]
    fn backward(_x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {
        *sum_grad += grad * res;
    }
}
struct Abs;
impl UnaryOpT for Abs {
    const OP: UnaryOp = UnaryOp::Abs;
    #[inline]
    fn forward(x: f64) -> f64 {
        x.abs()
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
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
    fn forward(x: f64) -> f64 {
        candle_core::cpu::erf::erf(x)
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
        let erf_grad = (2. / std::f64::consts::PI.sqrt()) * (-x * x).exp();
        *sum_grad += grad * erf_grad;
    }
}

struct LogicNot;
impl UnaryOpT for LogicNot {
    const OP: UnaryOp = UnaryOp::LogicNot;
    #[inline]
    fn debug_assertions(tensor: &Tensor) {
        assert_logic_tensor!(tensor);
    }
    #[inline]
    fn debug_mark(tensor: Tensor) -> Tensor {
        mark_logic_tensor!(tensor)
    }
    #[inline]
    fn forward(x: f64) -> f64 {
        assert_logic!(x);
        1.0 - x
    }
    #[inline]
    fn backward(x: &f64, _res: &f64, grad: &f64, sum_grad: &mut f64) {
        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
        let erf_grad = (2. / std::f64::consts::PI.sqrt()) * (-x * x).exp();
        *sum_grad += grad * erf_grad;
    }
}

impl UnaryOp {
    pub(super) const fn forward(&self) -> fn(f64) -> f64 {
        match self {
            Self::Neg => Neg::forward,
            Self::Sin => Sin::forward,
            Self::Cos => Cos::forward,
            Self::Tanh => Tanh::forward,
            Self::Tan => Tan::forward,
            Self::Ceil => Ceil::forward,
            Self::Floor => Floor::forward,
            Self::Round => Round::forward,
            Self::Sign => Sign::forward,
            Self::Sqrt => Sqrt::forward,
            Self::Sqr => Sqr::forward,
            Self::Cubic => Cubic::forward,
            Self::Log => Log::forward,
            Self::Exp => Exp::forward,
            Self::Abs => Abs::forward,
            Self::Erf => Erf::forward,
            Self::LogicNot => LogicNot::forward,
        }
    }
    #[inline]
    pub(super) const fn backward(&self) -> fn(&f64, &f64, &f64, &mut f64) {
        match self {
            Self::Neg => Neg::backward,
            Self::Sin => Sin::backward,
            Self::Cos => Cos::backward,
            Self::Tanh => Tanh::backward,
            Self::Tan => Tan::backward,
            Self::Ceil => Ceil::backward,
            Self::Floor => Floor::backward,
            Self::Round => Round::backward,
            Self::Sign => Sign::backward,
            Self::Sqrt => Sqrt::backward,
            Self::Sqr => Sqr::backward,
            Self::Cubic => Cubic::backward,
            Self::Log => Log::backward,
            Self::Exp => Exp::backward,
            Self::Abs => Abs::backward,
            Self::Erf => Erf::backward,
            Self::LogicNot => LogicNot::backward,
        }
    }
}

impl Tensor {
    #[inline]
    pub(super) fn iter_unary_op(&self, forward: fn(f64) -> f64) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .map(|x| forward(*x))
            .collect()
    }
    #[inline]
    pub(super) fn unary_op(&self, forward: fn(f64) -> f64, op: Op) -> Self {
        Self::new(
            if self.with_grad() {
                Some(GradId::new())
            } else {
                None
            },
            self.iter_unary_op(forward),
            op,
        )
    }
}

impl Expression {
    #[inline]
    pub fn neg(&self) -> Self {
        Self::unary_op::<Neg>(&self)
    }
    #[inline]
    pub fn sin(&self) -> Self {
        Self::unary_op::<Sin>(&self)
    }
    #[inline]
    pub fn cos(&self) -> Self {
        Self::unary_op::<Cos>(&self)
    }
    #[inline]
    pub fn tanh(&self) -> Self {
        Self::unary_op::<Tanh>(&self)
    }
    #[inline]
    pub fn tan(&self) -> Self {
        Self::unary_op::<Tan>(&self)
    }
    #[inline]
    pub fn ceil(&self) -> Self {
        Self::unary_op::<Ceil>(&self)
    }
    #[inline]
    pub fn floor(&self) -> Self {
        Self::unary_op::<Floor>(&self)
    }
    #[inline]
    pub fn round(&self) -> Self {
        Self::unary_op::<Round>(&self)
    }
    #[inline]
    pub fn sign(&self) -> Self {
        Self::unary_op::<Sign>(&self)
    }
    #[inline]
    pub fn sqrt(&self) -> Self {
        Self::unary_op::<Sqrt>(&self)
    }
    #[inline]
    pub fn sqr(&self) -> Self {
        Self::unary_op::<Sqr>(&self)
    }
    #[inline]
    pub fn cubic(&self) -> Self {
        Self::unary_op::<Cubic>(&self)
    }
    #[inline]
    pub fn log(&self) -> Self {
        Self::unary_op::<Log>(&self)
    }
    #[inline]
    pub fn exp(&self) -> Self {
        Self::unary_op::<Exp>(&self)
    }
    #[inline]
    pub fn abs(&self) -> Self {
        Self::unary_op::<Abs>(&self)
    }
    #[inline]
    pub fn erf(&self) -> Self {
        Self::unary_op::<Erf>(&self)
    }
    #[inline]
    pub fn logic_not(&self) -> Self {
        Self::unary_op::<LogicNot>(&self)
    }
}

impl Expression {
    #[inline]
    fn unary_op<T: UnaryOpT>(&self) -> Self {
        match self {
            Self::Const(x) => Self::Const(T::forward(*x)),
            Self::Tensor(tensor) => {
                T::debug_assertions(tensor);
                Self::Tensor(T::debug_mark(tensor.unary_op(
                    T::forward,
                    Op::Unary(Self::Tensor(tensor.clone()), T::OP),
                )))
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   DiscreteBinaryOp   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DiscreteBinaryOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

impl DiscreteBinaryOp {
    #[inline]
    pub(super) fn forward_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64)>,
    ) -> Vec<f64> {
        match self {
            DiscreteBinaryOp::Eq => Eq::forward_iter(iter),
            DiscreteBinaryOp::Ne => Ne::forward_iter(iter),
            DiscreteBinaryOp::Le => Le::forward_iter(iter),
            DiscreteBinaryOp::Ge => Ge::forward_iter(iter),
            DiscreteBinaryOp::Lt => Lt::forward_iter(iter),
            DiscreteBinaryOp::Gt => Gt::forward_iter(iter),
        }
    }
    #[inline]
    pub(super) fn forward_iter_fix_lhs<'a>(
        &self,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match self {
            DiscreteBinaryOp::Eq => Eq::forward_iter_fix_lhs(lhs, rhs_iter),
            DiscreteBinaryOp::Ne => Ne::forward_iter_fix_lhs(lhs, rhs_iter),
            DiscreteBinaryOp::Le => Le::forward_iter_fix_lhs(lhs, rhs_iter),
            DiscreteBinaryOp::Ge => Ge::forward_iter_fix_lhs(lhs, rhs_iter),
            DiscreteBinaryOp::Lt => Lt::forward_iter_fix_lhs(lhs, rhs_iter),
            DiscreteBinaryOp::Gt => Gt::forward_iter_fix_lhs(lhs, rhs_iter),
        }
    }
    #[inline]
    pub(super) fn forward_iter_fix_rhs<'a>(
        &self,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match self {
            DiscreteBinaryOp::Eq => Eq::forward_iter_fix_rhs(rhs, lhs_iter),
            DiscreteBinaryOp::Ne => Ne::forward_iter_fix_rhs(rhs, lhs_iter),
            DiscreteBinaryOp::Le => Le::forward_iter_fix_rhs(rhs, lhs_iter),
            DiscreteBinaryOp::Ge => Ge::forward_iter_fix_rhs(rhs, lhs_iter),
            DiscreteBinaryOp::Lt => Lt::forward_iter_fix_rhs(rhs, lhs_iter),
            DiscreteBinaryOp::Gt => Gt::forward_iter_fix_rhs(rhs, lhs_iter),
        }
    }
    #[inline]
    pub(super) fn backward_lhs_iter<'a>(
        &self,
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match self {
            DiscreteBinaryOp::Eq => Eq::backward_lhs_iter(grad_method, iter),
            DiscreteBinaryOp::Ne => Ne::backward_lhs_iter(grad_method, iter),
            DiscreteBinaryOp::Le => Le::backward_lhs_iter(grad_method, iter),
            DiscreteBinaryOp::Ge => Ge::backward_lhs_iter(grad_method, iter),
            DiscreteBinaryOp::Lt => Lt::backward_lhs_iter(grad_method, iter),
            DiscreteBinaryOp::Gt => Gt::backward_lhs_iter(grad_method, iter),
        }
    }
    #[inline]
    pub(super) fn backward_rhs_iter<'a>(
        &self,
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match self {
            DiscreteBinaryOp::Eq => Eq::backward_rhs_iter(grad_method, iter),
            DiscreteBinaryOp::Ne => Ne::backward_rhs_iter(grad_method, iter),
            DiscreteBinaryOp::Le => Le::backward_rhs_iter(grad_method, iter),
            DiscreteBinaryOp::Ge => Ge::backward_rhs_iter(grad_method, iter),
            DiscreteBinaryOp::Lt => Lt::backward_rhs_iter(grad_method, iter),
            DiscreteBinaryOp::Gt => Gt::backward_rhs_iter(grad_method, iter),
        }
    }
    #[inline]
    pub(super) fn backward_lhs_iter_fix_rhs<'a>(
        &self,
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match self {
            DiscreteBinaryOp::Eq => Eq::backward_lhs_iter_fix_rhs(grad_method, rhs, lhs_iter),
            DiscreteBinaryOp::Ne => Ne::backward_lhs_iter_fix_rhs(grad_method, rhs, lhs_iter),
            DiscreteBinaryOp::Le => Le::backward_lhs_iter_fix_rhs(grad_method, rhs, lhs_iter),
            DiscreteBinaryOp::Ge => Ge::backward_lhs_iter_fix_rhs(grad_method, rhs, lhs_iter),
            DiscreteBinaryOp::Lt => Lt::backward_lhs_iter_fix_rhs(grad_method, rhs, lhs_iter),
            DiscreteBinaryOp::Gt => Gt::backward_lhs_iter_fix_rhs(grad_method, rhs, lhs_iter),
        }
    }
    #[inline]
    pub(super) fn backward_rhs_iter_fix_lhs<'a>(
        &self,
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match self {
            DiscreteBinaryOp::Eq => Eq::backward_rhs_iter_fix_lhs(grad_method, lhs, rhs_iter),
            DiscreteBinaryOp::Ne => Ne::backward_rhs_iter_fix_lhs(grad_method, lhs, rhs_iter),
            DiscreteBinaryOp::Le => Le::backward_rhs_iter_fix_lhs(grad_method, lhs, rhs_iter),
            DiscreteBinaryOp::Ge => Ge::backward_rhs_iter_fix_lhs(grad_method, lhs, rhs_iter),
            DiscreteBinaryOp::Lt => Lt::backward_rhs_iter_fix_lhs(grad_method, lhs, rhs_iter),
            DiscreteBinaryOp::Gt => Gt::backward_rhs_iter_fix_lhs(grad_method, lhs, rhs_iter),
        }
    }
}

pub(super) trait GradMethodT: Debug + Clone {
    // TODO:
    fn sign_backward(&self, x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {}
    fn ceil_backward(&self, x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {}
    fn floor_backward(&self, x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {}
    fn round_backward(&self, x: &f64, res: &f64, grad: &f64, sum_grad: &mut f64) {}
    fn eq_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64);
    fn eq_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64);
    fn ne_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64);
    fn ne_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64);
    fn le_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64);
    fn le_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64);
    #[inline]
    fn ge_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        self.le_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad);
    }
    #[inline]
    fn ge_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        self.le_backward_lhs(lhs, rhs, res, grad, rhs_sum_grad);
    }
    fn lt_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64);
    fn lt_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64);
    #[inline]
    fn gt_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        self.lt_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad);
    }
    #[inline]
    fn gt_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        self.lt_backward_lhs(lhs, rhs, res, grad, rhs_sum_grad);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GradMethodDiscrete;
impl GradMethodT for GradMethodDiscrete {
    #[inline]
    fn eq_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        _ = (lhs, rhs, res, grad, lhs_sum_grad);
    }
    #[inline]
    fn eq_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        _ = (lhs, rhs, res, grad, rhs_sum_grad);
    }
    #[inline]
    fn ne_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        _ = (lhs, rhs, res, grad, lhs_sum_grad);
    }
    #[inline]
    fn ne_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        _ = (lhs, rhs, res, grad, rhs_sum_grad);
    }
    #[inline]
    fn le_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        _ = res;
        if OrderedFloat(*lhs).le(&OrderedFloat(*rhs)) {
            *lhs_sum_grad += grad;
        }
    }
    #[inline]
    fn le_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        _ = res;
        if OrderedFloat(*lhs).ge(&OrderedFloat(*rhs)) {
            *rhs_sum_grad += grad;
        }
    }
    #[inline]
    fn lt_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        _ = res;
        if OrderedFloat(*lhs).lt(&OrderedFloat(*rhs)) {
            *lhs_sum_grad += grad;
        }
    }
    #[inline]
    fn lt_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        _ = res;
        if OrderedFloat(*lhs).gt(&OrderedFloat(*rhs)) {
            *rhs_sum_grad += grad;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GradMethodLinear {
    epsilon: f64,
}

impl GradMethodT for GradMethodLinear {
    /// `1 - |a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    ///                1
    ///       /\       
    ///      /  \
    /// ____/    \___  0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    ///
    /// $$
    /// \frac{\partial \text{Eq}_{\text{linear}}}{\partial a} = \begin{cases}
    /// -\frac{\text{sign}(a - b)}{\epsilon} & \text{if } |a - b| < \epsilon \\
    /// 0 & \text{otherwise}
    /// \end{cases}
    /// $$
    #[inline]
    fn eq_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        if !res.is_zero() {
            *lhs_sum_grad -= grad * (lhs - rhs).signum() / self.epsilon;
        }
    }
    /// `1 - |a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    ///                1
    ///       /\       
    ///      /  \
    /// ____/    \___  0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    ///
    /// $$
    /// \frac{\partial \text{Eq}_{\text{linear}}}{\partial b} = \begin{cases}
    /// \frac{\text{sign}(a - b)}{\epsilon} & \text{if } |a - b| < \epsilon \\
    /// 0 & \text{otherwise}
    /// \end{cases}
    /// $$
    #[inline]
    fn eq_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        if !res.is_zero() {
            *rhs_sum_grad += grad * (lhs - rhs).signum() / self.epsilon;
        }
    }
    /// |`a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    /// ___      ____    1
    ///    \    /        
    ///     \  /
    ///      \/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    ///
    /// -eq
    #[inline]
    fn ne_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        if !res.is_one() {
            *lhs_sum_grad += grad * (lhs - rhs).signum() / self.epsilon;
        }
    }
    /// |`a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    /// ___      ____    1
    ///    \    /        
    ///     \  /
    ///      \/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    ///
    /// -eq
    #[inline]
    fn ne_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        if !res.is_one() {
            *rhs_sum_grad -= grad * (lhs - rhs).signum() / self.epsilon;
        }
    }
    /// `1/2 - (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    /// ____           1
    ///     \          
    ///       \
    ///         \___   0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    ///
    /// $$
    /// \frac{\partial \text{Lt}_{\text{linear}}}{\partial a} = \begin{cases}
    /// 0 & \text{if } |a - b| > \epsilon \\
    /// -\frac{1}{2\epsilon} & \text{if } |a - b| \leq \epsilon
    /// \end{cases}
    /// $$
    #[inline]
    fn le_backward_lhs(
        &self,
        lhs: &f64,
        rhs: &f64,
        _res: &f64,
        grad: &f64,
        lhs_sum_grad: &mut f64,
    ) {
        if OrderedFloat((lhs - rhs).abs()) <= OrderedFloat(self.epsilon) {
            *lhs_sum_grad -= grad / (2.0 * self.epsilon);
        }
    }
    /// `1/2 - (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    /// ____           1
    ///     \          
    ///       \
    ///         \___   0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    ///
    /// $$
    /// \frac{\partial \text{Lt}_{\text{linear}}}{\partial b} = \begin{cases}
    /// 0 & \text{if } |a - b| > \epsilon \\
    /// \frac{1}{2\epsilon} & \text{if } |a - b| \leq \epsilon
    /// \end{cases}
    /// $$
    #[inline]
    fn le_backward_rhs(
        &self,
        lhs: &f64,
        rhs: &f64,
        _res: &f64,
        grad: &f64,
        rhs_sum_grad: &mut f64,
    ) {
        if OrderedFloat((lhs - rhs).abs()) <= OrderedFloat(self.epsilon) {
            *rhs_sum_grad += grad / (2.0 * self.epsilon);
        }
    }
    fn lt_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        self.le_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad);
    }
    fn lt_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        self.le_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad);
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GradMethodSigmoid {
    k: f64,
}
impl GradMethodT for GradMethodSigmoid {
    /// `eq(a,b) = sigmoid(a, b, k) = e^(-k (a - b)^2)`
    ///
    /// $$ \frac{\partial \text{Eq}_{\text{sigmoid}}}{\partial a} = -2k (a - b) e^{-k (a - b)^2} $$
    #[inline]
    fn eq_backward_lhs(
        &self,
        lhs: &f64,
        rhs: &f64,
        _res: &f64,
        grad: &f64,
        lhs_sum_grad: &mut f64,
    ) {
        let diff = lhs - rhs;
        let kdiff = self.k * diff;
        *lhs_sum_grad -= grad * 2.0 * kdiff * ((-kdiff * diff).exp());
    }
    /// `eq(a,b) = sigmoid(a, b, k) = e^(-k (a - b)^2)`
    ///
    /// $$\frac{\partial \text{Eq}_{\text{sigmoid}}}{\partial b} = 2k (a - b) e^{-k (a - b)^2}$$
    #[inline]
    fn eq_backward_rhs(
        &self,
        lhs: &f64,
        rhs: &f64,
        _res: &f64,
        grad: &f64,
        rhs_sum_grad: &mut f64,
    ) {
        let diff = lhs - rhs;
        let kdiff = self.k * diff;
        *rhs_sum_grad += grad * 2.0 * kdiff * ((-kdiff * diff).exp());
    }
    /// `ne(a,b) = 1- sigmoid(a, b, k) = 1-e^(-k (a - b)^2)`
    ///
    /// -eq
    #[inline]
    fn ne_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        self.eq_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad);
    }
    /// `ne(a,b) = 1- sigmoid(a, b, k) = 1-e^(-k (a - b)^2)`
    ///
    /// -eq
    #[inline]
    fn ne_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        self.eq_backward_lhs(lhs, rhs, res, grad, rhs_sum_grad);
    }
    /// `le(a,b) = 1 / (1 + e^(k(a - b)))`
    ///
    /// $$\frac{\partial \text{Lt}_{\text{sigmoid}}}{\partial a} = -k \cdot \sigma(-k(a - b))(1 - \sigma(-k(a - b)))$$
    #[inline]
    fn le_backward_lhs(
        &self,
        lhs: &f64,
        rhs: &f64,
        _res: &f64,
        grad: &f64,
        lhs_sum_grad: &mut f64,
    ) {
        let sigma = 1.0 / (1.0 + (self.k * (lhs - rhs)).exp());
        *lhs_sum_grad -= grad * self.k * sigma * (1.0 - sigma);
    }
    /// `le(a,b) = 1 / (1 + e^(k(a - b)))`
    ///
    /// $$\frac{\partial \text{Lt}_{\text{sigmoid}}}{\partial b} = k \cdot \sigma(-k(a - b))(1 - \sigma(-k(a - b)))$$
    #[inline]
    fn le_backward_rhs(
        &self,
        lhs: &f64,
        rhs: &f64,
        _res: &f64,
        grad: &f64,
        rhs_sum_grad: &mut f64,
    ) {
        let sigma = 1.0 / (1.0 + (self.k * (lhs - rhs)).exp());
        *rhs_sum_grad += grad * self.k * sigma * (1.0 - sigma);
    }
    /// `lt(a,b) = 1 / (1 + e^(k(a - b)))`
    fn lt_backward_lhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        self.le_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad);
    }
    /// `lt(a,b) = 1 / (1 + e^(k(a - b)))`
    fn lt_backward_rhs(&self, lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        self.le_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad);
    }
}

pub(crate) trait DiscreteBinaryOpT {
    const OP: DiscreteBinaryOp;
    #[inline]
    fn debug_assertions(tensor: &Tensor) {
        _ = tensor;
    }
    #[inline]
    fn debug_mark(tensor: Tensor) -> Tensor {
        mark_logic_tensor!(tensor)
    }
    fn forward(lhs: f64, rhs: f64) -> f64;
    fn forward_iter<'a>(iter: impl Iterator<Item = (&'a f64, &'a f64)>) -> Vec<f64> {
        iter.map(|(lhs, rhs)| Self::forward(*lhs, *rhs)).collect()
    }
    #[inline]
    fn forward_iter_fix_lhs<'a>(lhs: f64, rhs_iter: impl Iterator<Item = &'a f64>) -> Vec<f64> {
        rhs_iter.map(move |rhs| Self::forward(lhs, *rhs)).collect()
    }
    #[inline]
    fn forward_iter_fix_rhs<'a>(rhs: f64, lhs_iter: impl Iterator<Item = &'a f64>) -> Vec<f64> {
        lhs_iter.map(move |lhs| Self::forward(*lhs, rhs)).collect()
    }
    fn backward_lhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    );
    fn backward_lhs_iter_fix_rhs<'a>(
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    );
    fn backward_rhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    );
    fn backward_rhs_iter_fix_lhs<'a>(
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    );
}

pub(super) struct Eq;
pub(super) struct Ne;
pub(super) struct Le;
pub(super) struct Ge;
pub(super) struct Lt;
pub(super) struct Gt;

impl Expression {
    #[inline]
    pub fn eq(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Eq>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn ne(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Ne>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn le(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Le>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn ge(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Ge>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn lt(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Lt>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn gt(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Gt>(rhs, GradMethod::Discrete)
    }
    /// `eq(a,b) = sigmoid(a, b, k) = e^(-k (a - b)^2)`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn eq_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Eq>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `ne(a,b) = 1- sigmoid(a, b, k) = 1-e^(-k (a - b)^2)`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn ne_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Ne>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `le(a,b) = 1 / (1 + e^(k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn le_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Le>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `ge(a,b) = 1 / (1 + e^(-k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn ge_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Ge>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `lt(a,b) = 1 / (1 + e^(k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn lt_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Lt>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `gt(a,b) = 1 / (1 + e^(-k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn gt_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Gt>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `1 - |a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    ///                1
    ///       /\       
    ///      /  \
    /// ____/    \___  0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn eq_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Eq>(rhs, GradMethod::new_linear(epsilon))
    }
    /// |`a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    /// ___      ____    1
    ///    \    /        
    ///     \  /
    ///      \/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn ne_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Ne>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 - (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    /// ____           1
    ///     \          
    ///       \
    ///         \___   0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn le_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Le>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 + (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    ///          ____  1
    ///         /      
    ///       /
    /// ____/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn ge_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Ge>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 - (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    /// ____           1
    ///     \          
    ///       \
    ///         \___   0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn lt_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Lt>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 + (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    ///          ____  1
    ///         /      
    ///       /
    /// ____/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn gt_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Gt>(rhs, GradMethod::new_linear(epsilon))
    }
}

impl Expression {
    /// GradMethod only activate in gradient mode
    #[inline]
    fn discrete_binary_op<T: DiscreteBinaryOpT>(
        &self,
        rhs: &Self,
        grad_method: GradMethod,
    ) -> Self {
        match (self, rhs) {
            (Self::Const(lhs_x), Self::Const(rhs_x)) => Self::Const(T::forward(*lhs_x, *rhs_x)),
            (Self::Const(lhs_x), Self::Tensor(rhs_tensor)) => {
                T::debug_assertions(rhs_tensor);
                let grad_id = if rhs_tensor.with_grad() {
                    Some(GradId::new())
                } else {
                    None
                };
                Self::Tensor(T::debug_mark(Tensor::new(
                    grad_id,
                    T::forward_iter_fix_lhs(*lhs_x, rhs_tensor.values().read().unwrap().iter()),
                    Op::DiscreteBinary(
                        Self::Const(*lhs_x),
                        Self::Tensor(rhs_tensor.clone()),
                        T::OP,
                        grad_method,
                    ),
                )))
            }
            (Self::Tensor(lhs_tensor), Self::Const(rhs_x)) => {
                T::debug_assertions(lhs_tensor);
                let grad_id = if lhs_tensor.with_grad() {
                    Some(GradId::new())
                } else {
                    None
                };
                Self::Tensor(T::debug_mark(Tensor::new(
                    grad_id,
                    T::forward_iter_fix_rhs(*rhs_x, lhs_tensor.values().read().unwrap().iter()),
                    Op::DiscreteBinary(
                        Self::Tensor(lhs_tensor.clone()),
                        Self::Const(*rhs_x),
                        T::OP,
                        grad_method,
                    ),
                )))
            }
            (Self::Tensor(lhs_tensor), Self::Tensor(rhs_tensor)) => {
                T::debug_assertions(lhs_tensor);
                T::debug_assertions(rhs_tensor);
                let grad_id = if lhs_tensor.with_grad() || rhs_tensor.with_grad() {
                    Some(GradId::new())
                } else {
                    None
                };
                Self::Tensor(T::debug_mark(Tensor::new(
                    grad_id,
                    T::forward_iter(izip!(
                        lhs_tensor.values().read().unwrap().iter(),
                        rhs_tensor.values().read().unwrap().iter()
                    )),
                    Op::DiscreteBinary(
                        Self::Tensor(lhs_tensor.clone()),
                        Self::Tensor(rhs_tensor.clone()),
                        T::OP,
                        grad_method,
                    ),
                )))
            }
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
    LogicAnd,
    LogicOr,
}

trait BinaryOpT {
    const OP: BinaryOp;
    #[inline]
    fn debug_assertions(tensor: &Tensor) {
        _ = tensor;
    }
    #[inline]
    fn debug_mark(tensor: Tensor) -> Tensor {
        tensor
    }
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64;
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64;
    fn backward_lhs(lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64);
    fn backward_rhs(lhs: &f64, rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64);
}

struct LogicAnd;
impl BinaryOpT for LogicAnd {
    const OP: BinaryOp = BinaryOp::LogicAnd;
    #[inline]
    fn debug_assertions(tensor: &Tensor) {
        assert_logic_tensor!(tensor);
    }
    #[inline]
    fn debug_mark(tensor: Tensor) -> Tensor {
        mark_logic_tensor!(tensor)
    }
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        assert_logic!(lhs);
        assert_logic!(rhs);
        lhs * rhs
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        assert_logic!(lhs);
        assert_logic!(rhs);
        lhs * rhs
    }
    #[inline]
    fn backward_lhs(_lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad * rhs;
    }
    #[inline]
    fn backward_rhs(lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad * lhs;
    }
}

/// or(a,b) = a+b - a * b
struct LogicOr;
impl BinaryOpT for LogicOr {
    const OP: BinaryOp = BinaryOp::LogicOr;
    #[inline]
    fn debug_assertions(tensor: &Tensor) {
        assert_logic_tensor!(tensor);
    }
    #[inline]
    fn debug_mark(tensor: Tensor) -> Tensor {
        mark_logic_tensor!(tensor)
    }
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        assert_logic!(lhs);
        assert_logic!(rhs);
        lhs + rhs - lhs * rhs
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        assert_logic!(lhs);
        assert_logic!(rhs);
        lhs + rhs - lhs * rhs
    }
    #[inline]
    fn backward_lhs(_lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad * (1.0 - rhs);
    }
    #[inline]
    fn backward_rhs(lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad * (1.0 - lhs);
    }
}

struct Add;
impl BinaryOpT for Add {
    const OP: BinaryOp = BinaryOp::Add;
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs + rhs
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs + rhs
    }
    #[inline]
    fn backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad;
    }
    #[inline]
    fn backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad;
    }
}
impl<'a, 'b> core::ops::Add<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn add(self, rhs: &'b Expression) -> Expression {
        self.add(rhs)
    }
}

struct Sub;
impl BinaryOpT for Sub {
    const OP: BinaryOp = BinaryOp::Sub;
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs - rhs
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs - rhs
    }
    #[inline]
    fn backward_lhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad;
    }
    #[inline]
    fn backward_rhs(_lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad -= grad;
    }
}
impl<'a, 'b> core::ops::Sub<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn sub(self, rhs: &'b Expression) -> Expression {
        self.sub(rhs)
    }
}

struct Mul;
impl BinaryOpT for Mul {
    const OP: BinaryOp = BinaryOp::Mul;
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs * rhs
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs * rhs
    }
    #[inline]
    fn backward_lhs(_lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad * rhs;
    }
    #[inline]
    fn backward_rhs(lhs: &f64, _rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad * lhs;
    }
}
impl<'a, 'b> core::ops::Mul<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn mul(self, rhs: &'b Expression) -> Expression {
        self.mul(rhs)
    }
}

struct Div;
impl BinaryOpT for Div {
    const OP: BinaryOp = BinaryOp::Div;
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs / rhs
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs / rhs
    }
    #[inline]
    fn backward_lhs(_lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad / rhs;
    }
    #[inline]
    fn backward_rhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad -= grad * lhs / (rhs * rhs);
    }
}
impl<'a, 'b> core::ops::Div<&'b Expression> for &'a Expression {
    type Output = Expression;
    #[inline]
    fn div(self, rhs: &'b Expression) -> Expression {
        self.div(rhs)
    }
}

struct Pow;
impl BinaryOpT for Pow {
    const OP: BinaryOp = BinaryOp::Pow;
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.powf(rhs)
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.powf(rhs)
    }
    /// $ c = a^b $
    ///
    /// $\frac{\partial f}{\partial a} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial a} = \frac{\partial f}{\partial c} \cdot b \cdot a^{b - 1}$
    ///
    #[inline]
    fn backward_lhs(lhs: &f64, rhs: &f64, res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        *lhs_sum_grad += grad * rhs * res / lhs;
        // *lhs_sum_grad += grad * rhs * lhs.powf(rhs - 1.0);
    }
    /// $\frac{\partial f}{\partial b} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial b} = \frac{\partial f}{\partial c} \cdot c \cdot \ln(a)$
    #[inline]
    fn backward_rhs(lhs: &f64, _rhs: &f64, res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
        *rhs_sum_grad += grad * res * (lhs.ln());
    }
}

struct Min;
impl BinaryOpT for Min {
    const OP: BinaryOp = BinaryOp::Min;
    #[inline]
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.min(rhs)
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.min(rhs)
    }
    #[inline]
    fn backward_lhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        // If both masks are 1 one the same point, we want to scale the
        // gradient by 0.5 rather than 1.
        match OrderedFloat(*lhs).cmp(&OrderedFloat(*rhs)) {
            Ordering::Less => *lhs_sum_grad += grad,
            Ordering::Equal => *lhs_sum_grad += grad / 2.0,
            Ordering::Greater => (),
        }
    }
    #[inline]
    fn backward_rhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
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
    fn forward_lhs_rhs(lhs: f64, rhs: f64) -> f64 {
        lhs.max(rhs)
    }
    #[inline]
    fn forward_rhs_lhs(rhs: f64, lhs: f64) -> f64 {
        lhs.max(rhs)
    }
    #[inline]
    fn backward_lhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, lhs_sum_grad: &mut f64) {
        // If both masks are 1 one the same point, we want to scale the
        // gradient by 0.5 rather than 1.
        match OrderedFloat(*lhs).cmp(&OrderedFloat(*rhs)) {
            Ordering::Less => (),
            Ordering::Equal => *lhs_sum_grad += grad / 2.0,
            Ordering::Greater => *lhs_sum_grad += grad,
        }
    }
    #[inline]
    fn backward_rhs(lhs: &f64, rhs: &f64, _res: &f64, grad: &f64, rhs_sum_grad: &mut f64) {
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
    pub(super) const fn forward(&self) -> [fn(f64, f64) -> f64; 2] {
        match self {
            Self::Add => [Add::forward_lhs_rhs, Add::forward_rhs_lhs],
            Self::Sub => [Sub::forward_lhs_rhs, Sub::forward_rhs_lhs],
            Self::Mul => [Mul::forward_lhs_rhs, Mul::forward_rhs_lhs],
            Self::Div => [Div::forward_lhs_rhs, Div::forward_rhs_lhs],
            Self::Pow => [Pow::forward_lhs_rhs, Pow::forward_rhs_lhs],
            Self::Min => [Min::forward_lhs_rhs, Min::forward_rhs_lhs],
            Self::Max => [Max::forward_lhs_rhs, Max::forward_rhs_lhs],
            Self::LogicAnd => [LogicAnd::forward_lhs_rhs, LogicAnd::forward_rhs_lhs],
            Self::LogicOr => [LogicOr::forward_lhs_rhs, LogicOr::forward_rhs_lhs],
        }
    }
    #[inline]
    pub(super) const fn backward(&self) -> [fn(&f64, &f64, &f64, &f64, &mut f64); 2] {
        match self {
            Self::Add => [Add::backward_lhs, Add::backward_rhs],
            Self::Sub => [Sub::backward_lhs, Sub::backward_rhs],
            Self::Mul => [Mul::backward_lhs, Mul::backward_rhs],
            Self::Div => [Div::backward_lhs, Div::backward_rhs],
            Self::Pow => [Pow::backward_lhs, Pow::backward_rhs],
            Self::Min => [Min::backward_lhs, Min::backward_rhs],
            Self::Max => [Max::backward_lhs, Max::backward_rhs],
            Self::LogicAnd => [LogicAnd::backward_lhs, LogicAnd::backward_rhs],
            Self::LogicOr => [LogicOr::backward_lhs, LogicOr::backward_rhs],
        }
    }
}

impl Tensor {
    #[inline]
    pub(super) fn iter_binary_op(&self, rhs: &Self, forward: fn(f64, f64) -> f64) -> Vec<f64> {
        let self_vec = self.values().read().unwrap();
        let rhs_vec = rhs.values().read().unwrap();
        debug_assert_eq!(rhs_vec.len(), self_vec.len(), "tensor length mismatch!");
        self_vec
            .iter()
            .zip(rhs_vec.iter())
            .map(|(v1, v2)| forward(*v1, *v2))
            .collect()
    }
    #[inline]
    pub(super) fn broadcast_iter_binary_op(
        &self,
        rhs: f64,
        forward: fn(f64, f64) -> f64,
    ) -> Vec<f64> {
        self.values()
            .read()
            .unwrap()
            .iter()
            .map(|v| forward(*v, rhs))
            .collect()
    }
    #[inline]
    pub(super) fn binary_op(&self, rhs: &Self, forward: fn(f64, f64) -> f64, op: Op) -> Self {
        Self::new(
            if self.with_grad() || rhs.with_grad() {
                Some(GradId::new())
            } else {
                None
            },
            self.iter_binary_op(rhs, forward),
            op,
        )
    }
    #[inline]
    pub(super) fn broadcast_binary_op(
        &self,
        rhs: f64,
        forward: fn(f64, f64) -> f64,
        op: Op,
    ) -> Self {
        Self::new(
            if self.with_grad() {
                Some(GradId::new())
            } else {
                None
            },
            self.broadcast_iter_binary_op(rhs, forward),
            op,
        )
    }
}

impl Expression {
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        self.binary_op::<Add>(rhs)
    }
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        self.binary_op::<Sub>(rhs)
    }
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        self.binary_op::<Mul>(rhs)
    }
    #[inline]
    pub fn div(&self, rhs: &Self) -> Self {
        self.binary_op::<Div>(rhs)
    }
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
    pub fn logic_and(&self, rhs: &Self) -> Self {
        self.binary_op::<LogicAnd>(rhs)
    }
    #[inline]
    pub fn logic_or(&self, rhs: &Self) -> Self {
        self.binary_op::<LogicOr>(rhs)
    }
}
impl Expression {
    #[inline]
    fn binary_op<T: BinaryOpT>(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Self::Const(lhs_x), Self::Const(rhs_x)) => {
                Self::Const(T::forward_lhs_rhs(*lhs_x, *rhs_x))
            }
            (Self::Const(lhs_x), Self::Tensor(rhs_tensor)) => {
                T::debug_assertions(rhs_tensor);
                Self::Tensor(T::debug_mark(rhs_tensor.broadcast_binary_op(
                    *lhs_x,
                    T::forward_rhs_lhs,
                    Op::Binary(Self::Const(*lhs_x), Self::Tensor(rhs_tensor.clone()), T::OP),
                )))
            }
            (Self::Tensor(lhs_tensor), Self::Const(rhs_x)) => {
                T::debug_assertions(lhs_tensor);
                Self::Tensor(T::debug_mark(lhs_tensor.broadcast_binary_op(
                    *rhs_x,
                    T::forward_lhs_rhs,
                    Op::Binary(Self::Tensor(lhs_tensor.clone()), Self::Const(*rhs_x), T::OP),
                )))
            }
            (Self::Tensor(lhs_tensor), Self::Tensor(rhs_tensor)) => {
                T::debug_assertions(lhs_tensor);
                T::debug_assertions(rhs_tensor);
                Self::Tensor(T::debug_mark(lhs_tensor.binary_op(
                    rhs_tensor,
                    T::forward_lhs_rhs,
                    Op::Binary(
                        Self::Tensor(lhs_tensor.clone()),
                        Self::Tensor(rhs_tensor.clone()),
                        T::OP,
                    ),
                )))
            }
        }
    }
}
