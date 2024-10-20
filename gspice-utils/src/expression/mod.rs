mod autograd;
mod impls;
mod op;
mod recompute;
mod test;

pub use recompute::before_update;

use autograd::GradId;
use op::Op;
use recompute::ChangeMarker;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug)]
pub struct Tensor(Arc<(Option<GradId>, RwLock<Vec<f64>>, ChangeMarker, Op)>);

#[derive(Clone, Debug)]
pub struct TensorRef(Tensor);

impl TensorRef {
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor = values
    #[inline]
    pub fn assgin(&self, values: Vec<f64>) {
        let mut write = self.0.values().write().unwrap();
        *write = values;
        self.0.change_marker().mark_searched_change();
    }
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor += delta
    #[inline]
    pub fn update(&self, delta: &[f64]) {
        self.update_callback(delta, |f| *f)
    }
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor += f(delta)
    #[inline]
    pub fn update_callback(&self, delta: &[f64], f: impl Fn(&f64) -> f64) {
        let mut write = self.0.values().write().unwrap();
        assert_eq!(write.len(), delta.len(), "tensor length mismatch!");
        write.iter_mut().zip(delta).for_each(|(x, d)| *x += f(d));
        self.0.change_marker().mark_searched_change();
    }
}

impl Tensor {
    #[inline]
    pub fn values(&self) -> &RwLock<Vec<f64>> {
        &self.0 .1
    }
    #[inline]
    pub fn with_grad(&self) -> bool {
        self.0 .0.is_some()
    }
    #[inline]
    fn zeros_like(&self) -> Vec<f64> {
        use num_traits::identities::Zero;
        vec![f64::zero(); self.values().read().unwrap().len()]
    }
    #[inline]
    fn ones_like(&self) -> Vec<f64> {
        use num_traits::identities::One;
        vec![f64::one(); self.values().read().unwrap().len()]
    }
    #[inline]
    fn op(&self) -> &Op {
        &self.0 .3
    }
    #[inline]
    fn grad_id(&self) -> &Option<GradId> {
        &self.0 .0
    }
    #[inline]
    fn change_marker(&self) -> &ChangeMarker {
        &self.0 .2
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
        let tensor = Tensor(Arc::new((
            if need_grad { Some(GradId::new()) } else { None },
            RwLock::new(values),
            ChangeMarker::new(),
            Op::Assgin,
        )));
        (Self::Tensor(tensor.clone()), TensorRef(tensor))
    }
    #[inline]
    pub fn zeros(len: usize, need_grad: bool) -> (Self, TensorRef) {
        use num_traits::identities::Zero;
        Self::tensor(vec![f64::zero(); len], need_grad)
    }
    #[inline]
    pub fn ones(len: usize, need_grad: bool) -> (Self, TensorRef) {
        use num_traits::identities::One;
        Self::tensor(vec![f64::one(); len], need_grad)
    }
    #[inline]
    pub fn rand<D: rand::distributions::Distribution<f64>>(
        len: usize,
        distr: D,
        need_grad: bool,
    ) -> (Self, TensorRef) {
        let mut rng = rand::thread_rng();
        Self::tensor(distr.sample_iter(&mut rng).take(len).collect(), need_grad)
    }
    #[inline]
    pub fn uniform(len: usize, lower: f64, upper: f64, need_grad: bool) -> (Self, TensorRef) {
        let distr = rand::distributions::Uniform::new(lower, upper);
        Self::rand(len, distr, need_grad)
    }
    /// get the value / recompute and get the value
    #[inline]
    pub fn value<'a>(&'a self) -> ScalarTensor<'a> {
        self.recompute().into()
    }
}
