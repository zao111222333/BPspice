#![cfg(test)]
use itertools::izip;
use ordered_float::OrderedFloat;
use serial_test::serial;

use crate::{before_update, Expression};
use std::ops::*;

use super::{autograd::Grad, ScalarTensor};

/// can NOT run test parallelly,
/// since the test functions will use global COUNTER
#[test]
#[serial]
fn test() {
    binary_op();
    unary_op();
    backward_mul_add();
    backward_pow();
    backward_min_max();
}

#[test]
#[serial]
fn gradient_decent() {
    let n = 200;
    let iter = 10000;
    let step = 0.01;
    let (x, x_ref) = Expression::uniform(n, -10.0, 10.0, true);
    let (y, y_ref) = Expression::uniform(n, -10.0, 10.0, true);
    let f = &x.mul(&x) + &y.mul(&y);
    let mut loss = f64::MAX;
    for i in 0..iter {
        if i % 200 == 0 {
            let new_loss = if let ScalarTensor::Tensor(tensor) = f.value() {
                tensor.read().unwrap().iter().fold(0.0, |sum,x|sum+x) / n as f64
            } else {
                unreachable!()
            };
            assert!(new_loss<loss);
            loss = new_loss;
            println!("iter {i}; loss = x^2+y^2 = {loss:5e}");
        }
        let grads = f.backward();
        let df_dx = grads.get(&x_ref).unwrap();
        let df_dy = grads.get(&y_ref).unwrap();
        before_update();
        x_ref.update_callback(&df_dx, |d: &f64| -d * step);
        y_ref.update_callback(&df_dy, |d: &f64| -d * step);
    }
    let loss = if let ScalarTensor::Tensor(tensor) = f.value() {
        tensor.read().unwrap().iter().fold(0.0, |sum,x|sum+x) / n as f64
    } else {
        unreachable!()
    };
    println!("iter {iter}; loss = x^2+y^2 = {loss:5e}");
}
#[test]
fn utils_ok() {
    assert_eq_vec(&[1.0, 2.0], &[1.0, 2.0]);
}

#[test]
#[should_panic]
fn utils_should_panic() {
    assert_eq_vec(&[1.0, 2.0], &[1.1, 2.0]);
}

#[test]
#[should_panic]
fn len_mismatch_init() {
    let (x, _) = Expression::tensor(vec![1.0, 2.0, 3.0], true);
    let (y, _) = Expression::tensor(vec![1.0, 2.0], true);
    _ = x.add(&y);
}

#[test]
#[serial]
#[should_panic]
fn len_mismatch_update() {
    let (x, x_ref) = Expression::tensor(vec![1.0, 2.0, 3.0], true);
    let (y, _) = Expression::tensor(vec![1.0, 2.0, 3.0], true);
    let f = x.add(&y);
    before_update();
    x_ref.assgin(vec![1.0]);
    _ = f.value();
}

fn assert_eq_vec(lhs: &[f64], rhs: &[f64]) {
    assert!(
        lhs.len() == rhs.len()
            && lhs
                .iter()
                .zip(rhs.iter())
                .all(|(x1, x2)| OrderedFloat(*x1).eq(&OrderedFloat(*x2))),
        "left:  {lhs:?}\nright: {rhs:?}"
    )
}

fn assert_grad(grad: Option<&Grad>, values: Vec<f64>) {
    if let Some(grad) = grad {
        assert_eq_vec(&grad, &values);
    } else {
        panic!("No grad");
    }
}

fn assert_ref(got: ScalarTensor<'_>, want: Vec<f64>) {
    match got {
        ScalarTensor::Tensor(tensor) => {
            assert_eq_vec(&tensor.read().unwrap(), &want);
        }
        _ => panic!("{got} is not tensor"),
    }
}

pub fn assert_scalar(got: ScalarTensor<'_>, want: f64) {
    match got {
        ScalarTensor::Scalar(f) => assert!(
            OrderedFloat(*f).eq(&OrderedFloat(want)),
            "left:  {f:?}\nright: {want:?}"
        ),
        _ => panic!("{got} is not number"),
    }
}

#[rustfmt::skip]
fn backward_mul_add() {
    let (a, a_ref) = Expression::tensor(vec![1.0, 2.0, 3.0], true);
    let (b, b_ref) = Expression::tensor(vec![-1.0, -2.0, -3.0], true);
    let (c, c_ref) = Expression::tensor(vec![4.0, -2.0, 9.0], true);
    let f = a.mul(&b).add(&c);
    let grads = f.backward();
    let df_da = grads.get(&a_ref);
    let df_db = grads.get(&b_ref);
    let df_dc = grads.get(&c_ref);
    assert_grad(df_da, vec![-1.0, -2.0, -3.0]);
    assert_grad(df_db, vec![1.0, 2.0, 3.0]);
    assert_grad(df_dc, vec![1.0, 1.0, 1.0]);

    // Update 1
    before_update();
    a_ref.assgin(vec![6.0]);
    b_ref.assgin(vec![-4.0]);
    c_ref.assgin(vec![2.0]);
    f.value();
    let grads = f.backward();
    let df_da = grads.get(&a_ref);
    let df_db = grads.get(&b_ref);
    let df_dc = grads.get(&c_ref);
    assert_grad(df_da, vec![-4.0]);
    assert_grad(df_db, vec![6.0]);
    assert_grad(df_dc, vec![1.0]);

    // Update 2
    before_update();
    a_ref.assgin(vec![2.0]);
    b_ref.assgin(vec![5.0]);
    c_ref.assgin(vec![2.0]);
    f.value();
    let grads = f.backward();
    let df_da = grads.get(&a_ref);
    let df_db = grads.get(&b_ref);
    let df_dc = grads.get(&c_ref);
    assert_grad(df_da, vec![5.0]);
    assert_grad(df_db, vec![2.0]);
    assert_grad(df_dc, vec![1.0]);
}

#[rustfmt::skip]
fn backward_pow() {
    let a_vec = vec![1.5, 2.0, 3.0];
    let b_vec = vec![3.0, 2.0, 4.0];
    let (a, a_ref) = Expression::tensor(a_vec.clone(), true);
    let (b, b_ref) = Expression::tensor(b_vec.clone(), true);
    let f = a.pow(&b);
    let grads = f.backward();
    let df_da = grads.get(&a_ref);
    let df_db = grads.get(&b_ref);
    assert_grad(df_da, izip!(a_vec.iter(),b_vec.iter()).map(|(a_x,b_x)|b_x*a_x.powf(b_x-1.0)).collect());
    assert_grad(df_db, izip!(a_vec.iter(),b_vec.iter()).map(|(a_x,b_x)|a_x.powf(*b_x)*a_x.ln()).collect());
}

#[rustfmt::skip]
fn backward_min_max() {
    let (a, a_ref) = Expression::tensor(vec![1.5, 2.0, 5.0], true);
    let (b, b_ref) = Expression::tensor(vec![3.0, 2.0, 4.0], true);
    let max = a.max(&b);
    let min = a.min(&b);
    let max_grads = max.backward();
    let min_grads = min.backward();
    let dmax_da = max_grads.get(&a_ref);
    let dmin_da = min_grads.get(&a_ref);
    let dmax_db = max_grads.get(&b_ref);
    let dmin_db = min_grads.get(&b_ref);
    assert_grad(dmax_da, vec![0.0, 0.5, 1.0]);
    assert_grad(dmin_da, vec![1.0, 0.5, 0.0]);
    assert_grad(dmax_db, vec![1.0, 0.5, 0.0]);
    assert_grad(dmin_db, vec![0.0, 0.5, 1.0]);
}

#[rustfmt::skip]
fn binary_op() {
    let const1 = Expression::constant(3.0);
    let const2 = Expression::constant(-2.0);
    let (tensor1, tensor1_ref) = Expression::tensor(vec![1.0, 2.0, 3.0], true);
    let (tensor2, tensor2_ref) = Expression::tensor(vec![-1.0, -2.0, -3.0], true);

    let const2_min_ref2 = const2.min(&tensor2);
    let const2_max_ref2 = const2.max(&tensor2);

    let const1_add_const2 = const1.add(&const2);
    let const1_sub_const2 = const1.sub(&const2);
    let const1_mul_const2 = const1.mul(&const2);
    let const1_div_const2 = const1.div(&const2);
    let const2_pow_const1 = const2.pow(&const1);

    let const1_add_ref1 = const1.add(&tensor1);
    let const1_sub_ref1 = const1.sub(&tensor1);
    let const1_mul_ref1 = const1.mul(&tensor1);
    let const1_div_ref1 = const1.div(&tensor1);
    let const1_pow_ref1 = const1.pow(&tensor1);

    let tensor1_add_const1 = tensor1.add(&const1);
    let tensor1_sub_const1 = tensor1.sub(&const1);
    let tensor1_mul_const1 = tensor1.mul(&const1);
    let tensor1_div_const1 = tensor1.div(&const1);
    let tensor1_pow_const1 = tensor1.pow(&const1);

    let tensor1_add_ref2 = tensor1.add(&tensor2);
    let tensor1_sub_ref2 = tensor1.sub(&tensor2);
    let tensor1_mul_ref2 = tensor1.mul(&tensor2);
    let tensor1_div_ref2 = tensor1.div(&tensor2);
    let tensor2_pow_ref1 = tensor2.pow(&tensor1);

    assert_ref(const2_max_ref2.value(), vec![-1.0, -2.0, -2.0]);
    assert_ref(const2_max_ref2.value(), vec![-1.0, -2.0, -2.0]);
    assert_ref(const2_min_ref2.value(), vec![-2.0, -2.0, -3.0]);

    assert_scalar(const1_add_const2.value(),1.0);
    assert_scalar(const1_sub_const2.value(),5.0);
    assert_scalar(const1_mul_const2.value(),-6.0);
    assert_scalar(const1_div_const2.value(),-1.5);
    assert_scalar(const2_pow_const1.value(),-8.0);

    assert_ref(const1_add_ref1.value(), vec![4.0, 5.0, 6.0]);
    assert_ref(const1_sub_ref1.value(), vec![2.0, 1.0, 0.0]);
    assert_ref(const1_mul_ref1.value(), vec![3.0, 6.0, 9.0]);
    assert_ref(const1_div_ref1.value(), vec![3.0, 1.5, 1.0]);
    assert_ref(const1_pow_ref1.value(), vec![3.0, 9.0, 27.0]);

    assert_ref(tensor1_add_const1.value(), vec![4.0, 5.0, 6.0]);
    assert_ref(tensor1_sub_const1.value(), vec![-2.0, -1.0, -0.0]);
    assert_ref(tensor1_mul_const1.value(), vec![3.0, 6.0, 9.0]);
    assert_ref(tensor1_div_const1.value(), vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
    assert_ref(tensor1_pow_const1.value(), vec![1.0, 8.0, 27.0]);

    assert_ref(tensor1_add_ref2.value(), vec![0.0, 0.0, 0.0]);
    assert_ref(tensor1_sub_ref2.value(), vec![2.0, 4.0, 6.0]);
    assert_ref(tensor1_mul_ref2.value(), vec![-1.0, -4.0, -9.0]);
    assert_ref(tensor1_div_ref2.value(), vec![-1.0, -1.0, -1.0]);
    assert_ref(tensor2_pow_ref1.value(), vec![-1.0, 4.0, -27.0]);
    // Update 1
    before_update();
    tensor1_ref.assgin(vec![-3.0, 6.0]);
    tensor2_ref.assgin(vec![3.0, -4.0]);

    assert_ref(const2_max_ref2.value(), vec![3.0, -2.0]);
    assert_ref(const2_min_ref2.value(), vec![-2.0, -4.0]);

    assert_scalar(const1_add_const2.value(),1.0);
    assert_scalar(const1_sub_const2.value(),5.0);
    assert_scalar(const1_mul_const2.value(),-6.0);
    assert_scalar(const1_div_const2.value(),-1.5);
    assert_scalar(const2_pow_const1.value(),-8.0);

    assert_ref(const1_add_ref1.value(), vec![0.0, 9.0]);
    assert_ref(const1_sub_ref1.value(), vec![6.0, -3.0]);
    assert_ref(const1_mul_ref1.value(), vec![-9.0, 18.0]);
    assert_ref(const1_div_ref1.value(), vec![-1.0, 0.5]);
    assert_ref(const1_pow_ref1.value(), vec![3.0_f64.powf(-3.0), 3.0_f64.powf(6.0)]);

    assert_ref(tensor1_add_const1.value(), vec![0.0, 9.0]);
    assert_ref(tensor1_sub_const1.value(), vec![-6.0, 3.0]);
    assert_ref(tensor1_mul_const1.value(), vec![-9.0, 18.0]);
    assert_ref(tensor1_div_const1.value(), vec![-1.0, 2.0]);
    assert_ref(tensor1_pow_const1.value(), vec![(-3.0_f64).powf(3.0), 6.0_f64.powf(3.0)]);

    assert_ref(tensor1_add_ref2.value(), vec![0.0, 2.0]);
    assert_ref(tensor1_sub_ref2.value(), vec![-6.0, 10.0]);
    assert_ref(tensor1_mul_ref2.value(), vec![-9.0, -24.0]);
    assert_ref(tensor1_div_ref2.value(), vec![-1.0, -1.5]);
    assert_ref(tensor2_pow_ref1.value(), vec![3.0_f64.powf(-3.0), (-4.0_f64).powf(6.0)]);

    // Update 2
    before_update();
    tensor1_ref.assgin(vec![6.0]);
    tensor2_ref.assgin(vec![-4.0]);

    assert_ref(const2_max_ref2.value(), vec![-2.0]);
    assert_ref(const2_min_ref2.value(), vec![-4.0]);

    assert_scalar(const1_add_const2.value(), 1.0);
    assert_scalar(const1_sub_const2.value(), 5.0);
    assert_scalar(const1_mul_const2.value(), -6.0);
    assert_scalar(const1_div_const2.value(), -1.5);
    assert_scalar(const2_pow_const1.value(), -8.0);

    assert_ref(const1_add_ref1.value(), vec![9.0]);
    assert_ref(const1_sub_ref1.value(), vec![-3.0]);
    assert_ref(const1_mul_ref1.value(), vec![18.0]);
    assert_ref(const1_div_ref1.value(), vec![0.5]);
    assert_ref(const1_pow_ref1.value(), vec![3.0_f64.powf(6.0)]);

    assert_ref(tensor1_add_const1.value(), vec![9.0]);
    assert_ref(tensor1_sub_const1.value(), vec![3.0]);
    assert_ref(tensor1_mul_const1.value(), vec![18.0]);
    assert_ref(tensor1_div_const1.value(), vec![2.0]);
    assert_ref(tensor1_pow_const1.value(), vec![6.0_f64.powf(3.0)]);

    assert_ref(tensor1_add_ref2.value(), vec![2.0]);
    assert_ref(tensor1_sub_ref2.value(), vec![10.0]);
    assert_ref(tensor1_mul_ref2.value(), vec![-24.0]);
    assert_ref(tensor1_div_ref2.value(), vec![-1.5]);
    assert_ref(tensor2_pow_ref1.value(), vec![(-4.0_f64).powf(6.0)]);
}

#[rustfmt::skip]
fn unary_op() {
    let values1 = vec![1.0, 2.0, 3.0];
    let x1 = 3.0;
    let const1 = Expression::constant(x1);
    let (tensor1, tensor1_ref) = Expression::tensor(values1.clone(), true);

    let tensor1_neg = tensor1.neg();
    let tensor1_sin = tensor1.sin();
    let tensor1_cos = tensor1.cos();
    let tensor1_tanh = tensor1.tanh();
    let tensor1_tan = tensor1.tan();
    let tensor1_ceil = tensor1.ceil();
    let tensor1_floor = tensor1.floor();
    let tensor1_round = tensor1.round();
    let tensor1_sign = tensor1.sign();
    let tensor1_sqrt = tensor1.sqrt();
    let tensor1_sqr = tensor1.sqr();
    let tensor1_log = tensor1.log();
    let tensor1_exp = tensor1.exp();
    let tensor1_abs = tensor1.abs();
    let tensor1_erf = tensor1.erf();

    let const1_neg = const1.neg();
    let const1_sin = const1.sin();
    let const1_cos = const1.cos();
    let const1_tanh = const1.tanh();
    let const1_tan = const1.tan();
    let const1_ceil = const1.ceil();
    let const1_floor = const1.floor();
    let const1_round = const1.round();
    let const1_sign = const1.sign();
    let const1_sqrt = const1.sqrt();
    let const1_sqr = const1.sqr();
    let const1_log = const1.log();
    let const1_exp = const1.exp();
    let const1_abs = const1.abs();
    let const1_erf = const1.erf();

    assert_ref(tensor1_neg.value(), values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>());
    assert_ref(tensor1_sin.value(), values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_cos.value(), values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_tanh.value(), values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_tan.value(), values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_ceil.value(), values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_floor.value(), values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_round.value(), values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sign.value(), values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sqrt.value(), values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sqr.value(), values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>());
    assert_ref(tensor1_log.value(), values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_exp.value(), values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_abs.value(), values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_erf.value(), values1.iter().map(|x| candle_core::cpu::erf::erf(*x)).collect::<Vec<_>>());

    assert_scalar(const1_neg.value(), Neg::neg(x1));
    assert_scalar(const1_sin.value(), f64::sin(x1));
    assert_scalar(const1_cos.value(), f64::cos(x1));
    assert_scalar(const1_tanh.value(), f64::tanh(x1));
    assert_scalar(const1_tan.value(), f64::tan(x1));
    assert_scalar(const1_ceil.value(), f64::ceil(x1));
    assert_scalar(const1_floor.value(), f64::floor(x1));
    assert_scalar(const1_round.value(), f64::round(x1));
    assert_scalar(const1_sign.value(), f64::signum(x1));
    assert_scalar(const1_sqrt.value(), f64::sqrt(x1));
    assert_scalar(const1_sqr.value(), f64::powi(x1, 2));
    assert_scalar(const1_log.value(), f64::ln(x1));
    assert_scalar(const1_exp.value(), f64::exp(x1));
    assert_scalar(const1_abs.value(), f64::abs(x1));
    assert_scalar(const1_erf.value(), candle_core::cpu::erf::erf(x1));

    // Update1
    let values1 = vec![1.0, 2.0];
    before_update();
    tensor1_ref.assgin(values1.clone());

    assert_ref(tensor1_neg.value(), values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>());
    assert_ref(tensor1_sin.value(), values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_cos.value(), values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_tanh.value(), values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_tan.value(), values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_ceil.value(), values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_floor.value(), values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_round.value(), values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sign.value(), values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sqrt.value(), values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sqr.value(), values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>());
    assert_ref(tensor1_log.value(), values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_exp.value(), values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_abs.value(), values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_erf.value(), values1.iter().map(|x| candle_core::cpu::erf::erf(*x)).collect::<Vec<_>>());

    assert_scalar(const1_neg.value(), Neg::neg(x1));
    assert_scalar(const1_sin.value(), f64::sin(x1));
    assert_scalar(const1_cos.value(), f64::cos(x1));
    assert_scalar(const1_tanh.value(), f64::tanh(x1));
    assert_scalar(const1_tan.value(), f64::tan(x1));
    assert_scalar(const1_ceil.value(), f64::ceil(x1));
    assert_scalar(const1_floor.value(), f64::floor(x1));
    assert_scalar(const1_round.value(), f64::round(x1));
    assert_scalar(const1_sign.value(), f64::signum(x1));
    assert_scalar(const1_sqrt.value(), f64::sqrt(x1));
    assert_scalar(const1_sqr.value(), f64::powi(x1, 2));
    assert_scalar(const1_log.value(), f64::ln(x1));
    assert_scalar(const1_exp.value(), f64::exp(x1));
    assert_scalar(const1_abs.value(), f64::abs(x1));
    assert_scalar(const1_erf.value(), candle_core::cpu::erf::erf(x1));

    // Update2
    let values1 = vec![1.0, 2.0];
    tensor1_ref.assgin(values1.clone());
    before_update();

    assert_ref(tensor1_neg.value(), values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>());
    assert_ref(tensor1_sin.value(), values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_cos.value(), values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_tanh.value(), values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_tan.value(), values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_ceil.value(), values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_floor.value(), values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_round.value(), values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sign.value(), values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sqrt.value(), values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_sqr.value(), values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>());
    assert_ref(tensor1_log.value(), values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_exp.value(), values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_abs.value(), values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>());
    assert_ref(tensor1_erf.value(), values1.iter().map(|x| candle_core::cpu::erf::erf(*x)).collect::<Vec<_>>());

    assert_scalar(const1_neg.value(), Neg::neg(x1));
    assert_scalar(const1_sin.value(), f64::sin(x1));
    assert_scalar(const1_cos.value(), f64::cos(x1));
    assert_scalar(const1_tanh.value(), f64::tanh(x1));
    assert_scalar(const1_tan.value(), f64::tan(x1));
    assert_scalar(const1_ceil.value(), f64::ceil(x1));
    assert_scalar(const1_floor.value(), f64::floor(x1));
    assert_scalar(const1_round.value(), f64::round(x1));
    assert_scalar(const1_sign.value(), f64::signum(x1));
    assert_scalar(const1_sqrt.value(), f64::sqrt(x1));
    assert_scalar(const1_sqr.value(), f64::powi(x1, 2));
    assert_scalar(const1_log.value(), f64::ln(x1));
    assert_scalar(const1_exp.value(), f64::exp(x1));
    assert_scalar(const1_abs.value(), f64::abs(x1));
    assert_scalar(const1_erf.value(), candle_core::cpu::erf::erf(x1));
}
