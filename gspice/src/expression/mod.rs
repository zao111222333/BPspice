mod autograd;
mod impls;
mod op;
mod recompute;

use autograd::GradId;
pub use op::Op;
pub use recompute::ChangeMarker;

use std::sync::{Arc, RwLock};



#[derive(Clone, Debug)]
pub struct Tensor(Arc<(Option<GradId>, RwLock<Vec<f64>>, ChangeMarker)>);

impl Tensor {
    pub fn update(&self, values: Vec<f64>) {
        let mut write = self.values().write().unwrap();
        *write = values;
        self.change_marker().mark_searched_change();
    }
    fn grad_id(&self) -> &Option<GradId> {
        &self.0 .0
    }
    fn values(&self) -> &RwLock<Vec<f64>> {
        &self.0 .1
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
    pub fn value<'a>(&'a self) -> ScalarTensor<'a> {
        match &self {
            Self::Const(f) => ScalarTensor::Scalar(f),
            Self::Parameter(tensor) | Self::Operation(tensor, _) => ScalarTensor::Tensor(tensor),
        }
    }
    pub fn parameter(values: Vec<f64>, need_grad: bool) -> (Self, Tensor) {
        let tensor = Tensor(Arc::new((
            if need_grad { Some(GradId::new()) } else { None },
            RwLock::new(values),
            ChangeMarker::new(),
        )));
        (Self::Parameter(tensor.clone()), tensor)
    }
    pub fn constant(value: f64) -> Self {
        Self::Const(value)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ops::*;
    #[test]
    fn binary_op() {
        let const1 = Expression::constant(3.0);
        let const2 = Expression::constant(-2.0);
        let (param1, param1_tensor) = Expression::parameter(vec![1.0, 2.0, 3.0], true);
        let (param2, param2_tensor) = Expression::parameter(vec![-1.0, -2.0, -3.0], true);

        let const2_min_param2 = const2.min(&param2);
        let const2_max_param2 = const2.max(&param2);

        let const1_add_const2 = const1.add(&const2);
        let const1_sub_const2 = const1.sub(&const2);
        let const1_mul_const2 = const1.mul(&const2);
        let const1_div_const2 = const1.div(&const2);
        let const2_pow_const1 = const2.pow(&const1);

        let const1_add_param1 = const1.add(&param1);
        let const1_sub_param1 = const1.sub(&param1);
        let const1_mul_param1 = const1.mul(&param1);
        let const1_div_param1 = const1.div(&param1);
        let const1_pow_param1 = const1.pow(&param1);

        let param1_add_const1 = param1.add(&const1);
        let param1_sub_const1 = param1.sub(&const1);
        let param1_mul_const1 = param1.mul(&const1);
        let param1_div_const1 = param1.div(&const1);
        let param1_pow_const1 = param1.pow(&const1);

        let param1_add_param2 = param1.add(&param2);
        let param1_sub_param2 = param1.sub(&param2);
        let param1_mul_param2 = param1.mul(&param2);
        let param1_div_param2 = param1.div(&param2);
        let param2_pow_param1 = param2.pow(&param1);

        assert!(const2_max_param2.eq_vec(&vec![-1.0, -2.0, -2.0]));
        assert!(const2_min_param2.eq_vec(&vec![-2.0, -2.0, -3.0]));

        assert!(const1_add_const2.eq_num(1.0));
        assert!(const1_sub_const2.eq_num(5.0));
        assert!(const1_mul_const2.eq_num(-6.0));
        assert!(const1_div_const2.eq_num(-1.5));
        assert!(const2_pow_const1.eq_num(-8.0));

        assert!(const1_add_param1.eq_vec(&vec![4.0, 5.0, 6.0]));
        assert!(const1_sub_param1.eq_vec(&vec![2.0, 1.0, 0.0]));
        assert!(const1_mul_param1.eq_vec(&vec![3.0, 6.0, 9.0]));
        assert!(const1_div_param1.eq_vec(&vec![3.0, 1.5, 1.0]));
        assert!(const1_pow_param1.eq_vec(&vec![3.0, 9.0, 27.0]));

        assert!(param1_add_const1.eq_vec(&vec![4.0, 5.0, 6.0]));
        assert!(param1_sub_const1.eq_vec(&vec![-2.0, -1.0, -0.0]));
        assert!(param1_mul_const1.eq_vec(&vec![3.0, 6.0, 9.0]));
        assert!(param1_div_const1.eq_vec(&vec![1.0 / 3.0, 2.0 / 3.0, 1.0]));
        assert!(param1_pow_const1.eq_vec(&vec![1.0, 8.0, 27.0]));

        assert!(param1_add_param2.eq_vec(&vec![0.0, 0.0, 0.0]));
        assert!(param1_sub_param2.eq_vec(&vec![2.0, 4.0, 6.0]));
        assert!(param1_mul_param2.eq_vec(&vec![-1.0, -4.0, -9.0]));
        assert!(param1_div_param2.eq_vec(&vec![-1.0, -1.0, -1.0]));
        assert!(param2_pow_param1.eq_vec(&vec![-1.0, 4.0, -27.0]));

        // Update 1
        param1_tensor.update(vec![-3.0, 6.0]);
        param2_tensor.update(vec![3.0, -4.0]);

        const2_max_param2.recompute();
        const2_min_param2.recompute();

        const1_add_const2.recompute();
        const1_sub_const2.recompute();
        const1_mul_const2.recompute();
        const1_div_const2.recompute();
        const2_pow_const1.recompute();

        const1_add_param1.recompute();
        const1_sub_param1.recompute();
        const1_mul_param1.recompute();
        const1_div_param1.recompute();
        const1_pow_param1.recompute();

        param1_add_const1.recompute();
        param1_sub_const1.recompute();
        param1_mul_const1.recompute();
        param1_div_const1.recompute();
        param1_pow_const1.recompute();

        param1_add_param2.recompute();
        param1_sub_param2.recompute();
        param1_mul_param2.recompute();
        param1_div_param2.recompute();
        param2_pow_param1.recompute();

        ChangeMarker::recompute_done();

        assert!(const2_max_param2.eq_vec(&vec![3.0, -2.0]));
        assert!(const2_min_param2.eq_vec(&vec![-2.0, -4.0]));

        assert!(const1_add_const2.eq_num(1.0));
        assert!(const1_sub_const2.eq_num(5.0));
        assert!(const1_mul_const2.eq_num(-6.0));
        assert!(const1_div_const2.eq_num(-1.5));
        assert!(const2_pow_const1.eq_num(-8.0));

        assert!(const1_add_param1.eq_vec(&vec![0.0, 9.0]));
        assert!(const1_sub_param1.eq_vec(&vec![6.0, -3.0]));
        assert!(const1_mul_param1.eq_vec(&vec![-9.0, 18.0]));
        assert!(const1_div_param1.eq_vec(&vec![-1.0, 0.5]));
        assert!(const1_pow_param1.eq_vec(&vec![3.0_f64.powf(-3.0), 3.0_f64.powf(6.0)]));

        assert!(param1_add_const1.eq_vec(&vec![0.0, 9.0]));
        assert!(param1_sub_const1.eq_vec(&vec![-6.0, 3.0]));
        assert!(param1_mul_const1.eq_vec(&vec![-9.0, 18.0]));
        assert!(param1_div_const1.eq_vec(&vec![-1.0, 2.0]));
        assert!(param1_pow_const1.eq_vec(&vec![(-3.0_f64).powf(3.0), 6.0_f64.powf(3.0)]));

        assert!(param1_add_param2.eq_vec(&vec![0.0, 2.0]));
        assert!(param1_sub_param2.eq_vec(&vec![-6.0, 10.0]));
        assert!(param1_mul_param2.eq_vec(&vec![-9.0, -24.0]));
        assert!(param1_div_param2.eq_vec(&vec![-1.0, -1.5]));
        assert!(param2_pow_param1.eq_vec(&vec![3.0_f64.powf(-3.0), (-4.0_f64).powf(6.0)]));

        // Update 2
        param1_tensor.update(vec![6.0]);
        param2_tensor.update(vec![-4.0]);

        const2_max_param2.recompute();
        const2_min_param2.recompute();

        const1_add_const2.recompute();
        const1_sub_const2.recompute();
        const1_mul_const2.recompute();
        const1_div_const2.recompute();
        const2_pow_const1.recompute();

        const1_add_param1.recompute();
        const1_sub_param1.recompute();
        const1_mul_param1.recompute();
        const1_div_param1.recompute();
        const1_pow_param1.recompute();

        param1_add_const1.recompute();
        param1_sub_const1.recompute();
        param1_mul_const1.recompute();
        param1_div_const1.recompute();
        param1_pow_const1.recompute();

        param1_add_param2.recompute();
        param1_sub_param2.recompute();
        param1_mul_param2.recompute();
        param1_div_param2.recompute();
        param2_pow_param1.recompute();

        ChangeMarker::recompute_done();

        assert!(const2_max_param2.eq_vec(&vec![-2.0]));
        assert!(const2_min_param2.eq_vec(&vec![-4.0]));

        assert!(const1_add_const2.eq_num(1.0));
        assert!(const1_sub_const2.eq_num(5.0));
        assert!(const1_mul_const2.eq_num(-6.0));
        assert!(const1_div_const2.eq_num(-1.5));
        assert!(const2_pow_const1.eq_num(-8.0));

        assert!(const1_add_param1.eq_vec(&vec![9.0]));
        assert!(const1_sub_param1.eq_vec(&vec![-3.0]));
        assert!(const1_mul_param1.eq_vec(&vec![18.0]));
        assert!(const1_div_param1.eq_vec(&vec![0.5]));
        assert!(const1_pow_param1.eq_vec(&vec![3.0_f64.powf(6.0)]));

        assert!(param1_add_const1.eq_vec(&vec![9.0]));
        assert!(param1_sub_const1.eq_vec(&vec![3.0]));
        assert!(param1_mul_const1.eq_vec(&vec![18.0]));
        assert!(param1_div_const1.eq_vec(&vec![2.0]));
        assert!(param1_pow_const1.eq_vec(&vec![6.0_f64.powf(3.0)]));

        assert!(param1_add_param2.eq_vec(&vec![2.0]));
        assert!(param1_sub_param2.eq_vec(&vec![10.0]));
        assert!(param1_mul_param2.eq_vec(&vec![-24.0]));
        assert!(param1_div_param2.eq_vec(&vec![-1.5]));
        assert!(param2_pow_param1.eq_vec(&vec![(-4.0_f64).powf(6.0)]));
    }
}
