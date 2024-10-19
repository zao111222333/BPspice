mod autograd;
mod impls;
mod op;
mod recompute;

pub use recompute::before_update;

use op::Op;
use autograd::GradId;
use recompute::ChangeMarker;

use std::sync::{Arc, RwLock};

#[derive(Clone, Debug)]
pub struct Tensor(Arc<(Option<GradId>, RwLock<Vec<f64>>, ChangeMarker)>);

impl Tensor {
    /// Need [`before_update`] before call this
    pub fn update(&self, values: Vec<f64>) {
        let mut write = self.values().write().unwrap();
        *write = values;
        self.change_marker().mark_searched_change();
    }
    pub fn values(&self) -> &RwLock<Vec<f64>> {
        &self.0 .1
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
    /// Parameter could be modified, e.g., swipe
    /// Parameter could need gradient
    Parameter(Tensor),
    Operation(Tensor, Arc<Op>),
}

#[derive(Clone, Debug)]
pub enum ScalarTensor<'a> {
    Scalar(&'a f64),
    Tensor(&'a Tensor),
}

impl Expression {
    pub fn constant(value: f64) -> Self {
        Self::Const(value)
    }
    pub fn parameter(values: Vec<f64>, need_grad: bool) -> (Self, Tensor) {
        let tensor = Tensor(Arc::new((
            if need_grad { Some(GradId::new()) } else { None },
            RwLock::new(values),
            ChangeMarker::new(),
        )));
        (Self::Parameter(tensor.clone()), tensor)
    }
    pub fn value<'a>(&'a self) -> ScalarTensor<'a> {
        self.recompute().into()
    }
}
