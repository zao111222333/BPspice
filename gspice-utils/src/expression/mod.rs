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
    pub fn update(&self, delta: &[f64]) {
        let mut write = self.0.values().write().unwrap();
        assert_eq!(write.len(), delta.len());
        write.iter_mut().zip(delta).for_each(|(x, d)| *x += d);
        self.0.change_marker().mark_searched_change();
    }
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor += f(delta)
    pub fn update_callback(&self, delta: &[f64], f: impl Fn(&f64) -> f64) {
        let mut write = self.0.values().write().unwrap();
        assert_eq!(write.len(), delta.len());
        write.iter_mut().zip(delta).for_each(|(x, d)| *x += f(d));
        self.0.change_marker().mark_searched_change();
    }
}

impl Tensor {
    pub fn values(&self) -> &RwLock<Vec<f64>> {
        &self.0 .1
    }
    pub fn with_grad(&self) -> bool {
        self.0 .0.is_some()
    }
    fn zeros_like(&self) -> Vec<f64> {
        vec![0.0; self.values().read().unwrap().len()]
    }
    fn ones_like(&self) -> Vec<f64> {
        vec![1.0; self.values().read().unwrap().len()]
    }
    fn op(&self) -> &Op {
        &self.0 .3
    }
    fn grad_id(&self) -> &Option<GradId> {
        &self.0 .0
    }
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
    pub fn constant(value: f64) -> Self {
        Self::Const(value)
    }
    pub fn tensor(values: Vec<f64>, need_grad: bool) -> (Self, TensorRef) {
        let tensor = Tensor(Arc::new((
            if need_grad { Some(GradId::new()) } else { None },
            RwLock::new(values),
            ChangeMarker::new(),
            Op::Assgin,
        )));
        (Self::Tensor(tensor.clone()), TensorRef(tensor))
    }
    /// get the value / recompute and get the value
    pub fn value<'a>(&'a self) -> ScalarTensor<'a> {
        self.recompute().into()
    }
}
