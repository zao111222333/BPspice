mod autograd;
mod impls;
mod op;
mod optimizer;
mod recompute;
mod test;
use itertools::zip_eq;
pub use recompute::before_update;

use autograd::GradId;
use num_traits::identities::{One, Zero};
use op::Op;
use recompute::ChangeMarker;
use std::sync::{
    atomic::{AtomicBool, Ordering::Relaxed},
    Arc, RwLock,
};

#[derive(Clone, Debug)]
pub struct Tensor(Arc<_Tensor>);

#[derive(Debug)]
struct _Tensor {
    grad_id: Option<GradId>,
    values: RwLock<Vec<f64>>,
    change_marker: ChangeMarker,
    op: Op,
    #[cfg(debug_assertions)]
    is_logic: AtomicBool,
}
impl Tensor {
    #[inline]
    pub fn values(&self) -> &RwLock<Vec<f64>> {
        &self.0.values
    }
    #[inline]
    pub fn with_grad(&self) -> bool {
        self.0.grad_id.is_some()
    }
    #[inline]
    fn zeros_like(&self) -> Vec<f64> {
        vec![f64::zero(); self.values().read().unwrap().len()]
    }
    #[inline]
    fn ones_like(&self) -> Vec<f64> {
        vec![f64::one(); self.values().read().unwrap().len()]
    }
    #[inline]
    fn op(&self) -> &Op {
        &self.0.op
    }
    #[inline]
    fn grad_id(&self) -> &Option<GradId> {
        &self.0.grad_id
    }
    #[inline]
    fn change_marker(&self) -> &ChangeMarker {
        &self.0.change_marker
    }
    #[cfg(debug_assertions)]
    #[inline]
    fn is_logic(&self) -> bool {
        self.0.is_logic.load(Relaxed)
    }
    #[cfg(debug_assertions)]
    #[inline]
    fn mark_logic(&self) {
        self.0.is_logic.store(true, Relaxed)
    }
    #[inline]
    fn new(grad_id: Option<GradId>, values: Vec<f64>, op: Op) -> Self {
        Self(Arc::new(_Tensor {
            grad_id,
            values: RwLock::new(values),
            change_marker: ChangeMarker::new(),
            op,
            #[cfg(debug_assertions)]
            is_logic: AtomicBool::new(false),
        }))
    }
}

#[derive(Clone, Debug)]
pub struct TensorRef(Tensor);

impl TensorRef {
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor = values
    #[inline]
    pub fn assign(&self, values: Vec<f64>) {
        let mut write = self.0.values().write().unwrap();
        *write = values;
        self.0.change_marker().mark_searched_change();
    }
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor\[i\] += delta\[i\]
    #[inline]
    pub fn update(&self, delta: &[f64]) {
        self.update_iter(delta.into_iter().map(|d|*d))
    }
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor\[i\] += delta_iter\[i\]
    #[inline]
    pub fn update_iter(&self, delta_iter: impl Iterator<Item = f64>) {
        let mut write = self.0.values().write().unwrap();
        zip_eq(write.iter_mut(), delta_iter).for_each(|(x, d)| *x += d);
        self.0.change_marker().mark_searched_change();
    }
}

#[derive(Clone, Debug)]
pub enum Expression {
    Const(f64),
    /// Tensor could be modified, e.g., swipe
    /// Tensor could need gradient
    Tensor(Tensor),
}

#[derive(Clone, Debug)]
pub enum ScalarTensor<'a> {
    Scalar(&'a f64),
    Tensor(&'a RwLock<Vec<f64>>),
}

impl Expression {
    #[inline]
    pub fn constant(value: f64) -> Self {
        Self::Const(value)
    }
    #[inline]
    pub fn tensor(values: Vec<f64>, need_grad: bool) -> (Self, TensorRef) {
        let tensor = Tensor::new(
            if need_grad { Some(GradId::new()) } else { None },
            values,
            Op::Assgin,
        );
        (Self::Tensor(tensor.clone()), TensorRef(tensor))
    }
    #[inline]
    pub fn zeros(len: usize, need_grad: bool) -> (Self, TensorRef) {
        Self::tensor(vec![f64::zero(); len], need_grad)
    }
    #[inline]
    pub fn ones(len: usize, need_grad: bool) -> (Self, TensorRef) {
        Self::tensor(vec![f64::one(); len], need_grad)
    }
    #[inline]
    pub fn rand<T, D: rand::distributions::Distribution<T>>(
        len: usize,
        distr: D,
        f: fn(T) -> f64,
        need_grad: bool,
    ) -> (Self, TensorRef) {
        let mut rng = rand::thread_rng();
        Self::tensor(
            distr.sample_iter(&mut rng).take(len).map(f).collect(),
            need_grad,
        )
    }
    #[inline]
    pub fn rand_uniform(len: usize, lower: f64, upper: f64, need_grad: bool) -> (Self, TensorRef) {
        let distr = rand::distributions::Uniform::new(lower, upper);
        Self::rand(len, distr, |f| f, need_grad)
    }
    #[inline]
    pub fn rand_bernoulli(len: usize, p: f64, need_grad: bool) -> (Self, TensorRef) {
        let distr =
            rand::distributions::Bernoulli::new(p.max(f64::zero()).min(f64::one())).unwrap();
        Self::rand(
            len,
            distr,
            |b| if b { f64::one() } else { f64::zero() },
            need_grad,
        )
    }
    /// get the value / recompute and get the value
    #[inline]
    pub fn value<'a>(&'a self) -> ScalarTensor<'a> {
        self.recompute().into()
    }
    /// Mark the expression as logic for debug-mode-only logic check
    ///
    /// `#[cfg(test)]` This requirement seems only happend in test
    #[cfg(test)]
    pub fn mark_logic(&self) {
        #[cfg(debug_assertions)]
        match self {
            Expression::Const(_) => todo!(),
            Expression::Tensor(tensor) => tensor.mark_logic(),
        }
    }
}
