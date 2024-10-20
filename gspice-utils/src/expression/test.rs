#![cfg(test)]
use itertools::izip;
use ordered_float::OrderedFloat;
use serial_test::serial;

use crate::{before_update, Expression};
use std::ops::*;

use super::ScalarTensor;

macro_rules! assert_eq_vec {
    ($lhs:expr, $rhs:expr) => {
        let lhs = $lhs;
        let rhs = $rhs;
        assert!(
            lhs.len() == rhs.len()
                && lhs
                    .iter()
                    .zip(rhs.iter())
                    .all(|(x1, x2)| OrderedFloat(*x1).eq(&OrderedFloat(*x2))),
            "left:  {lhs:?}\nright: {rhs:?}"
        )
    };
}

macro_rules! assert_grad {
    ($got:expr, $want:expr) => {
        let got = $got;
        let want: Vec<f64> = $want;
        if let Some(got) = got {
            assert_eq_vec!(&got, &want);
        } else {
            panic!("No grad");
        }
    };
}

macro_rules! assert_candle_scalar {
    ($got:expr, $want:expr) => {
        assert_eq!(OrderedFloat($want.to_vec0::<f64>().unwrap()), OrderedFloat($got.value().to_scalar().unwrap()));
    };
}

macro_rules! assert_candle_tensor {
    ($got:expr, $want:expr) => {
        assert_eq_vec!($want.to_vec1::<f64>().unwrap(), $got.value().to_tensor().unwrap());
    };
    ($got:expr, $want:expr, ($got_tensor1:expr, $want_tensor1:expr)) => {
        assert_candle_tensor!($got, $want);
        let (grads, grads_candle) = ($got.backward(), $want.backward().unwrap());
        assert_eq_vec!(&grads.get($got_tensor1).unwrap(), &grads_candle.get($want_tensor1).unwrap().to_vec1::<f64>().unwrap());    
    };
    ($got:expr, $want:expr, ($got_tensor1:expr, $want_tensor1:expr), ($got_tensor2:expr, $want_tensor2:expr)) => {
        assert_candle_tensor!($got, $want);
        let (grads, grads_candle) = ($got.backward(), $want.backward().unwrap());
        assert_eq_vec!(&grads.get($got_tensor1).unwrap(), &grads_candle.get($want_tensor1).unwrap().to_vec1::<f64>().unwrap());    
        assert_eq_vec!(&grads.get($got_tensor2).unwrap(), &grads_candle.get($want_tensor2).unwrap().to_vec1::<f64>().unwrap());    
    };
}

macro_rules! assert_tensor {
    ($got:expr, $want:expr) => {
        let got = $got;
        let want: Vec<f64> = $want;
        match got.value() {
            ScalarTensor::Tensor(tensor) => {
                assert_eq_vec!(&tensor.read().unwrap(), &want);
            }
            _ => panic!("{got} is not tensor"),
        }
    };
}

macro_rules! assert_scalar {
    ($got:expr, $want:expr) => {
        let got = $got;
        let want: f64 = $want;
        match got.value() {
            ScalarTensor::Scalar(f) => assert!(
                OrderedFloat(*f).eq(&OrderedFloat(want)),
                "left:  {f:?}\nright: {want:?}"
            ),
            _ => panic!("{got} is not number"),
        }
    };
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
            let new_loss = f
                .value()
                .to_tensor()
                .unwrap()
                .iter()
                .fold(0.0, |sum, x| sum + x)
                / n as f64;
            assert!(new_loss < loss);
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
    let loss = f
        .value()
        .to_tensor()
        .unwrap()
        .iter()
        .fold(0.0, |sum, x| sum + x)
        / n as f64;
    println!("iter {iter}; loss = x^2+y^2 = {loss:5e}");
}
#[test]
fn utils_ok() {
    assert_eq_vec!(&[1.0, 2.0], &[1.0, 2.0]);
}

#[test]
#[should_panic]
fn utils_should_panic() {
    assert_eq_vec!(&[1.0, 2.0], &[1.1, 2.0]);
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

#[test]
#[serial]
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
    assert_grad!(df_da, vec![-1.0, -2.0, -3.0]);
    assert_grad!(df_db, vec![1.0, 2.0, 3.0]);
    assert_grad!(df_dc, vec![1.0, 1.0, 1.0]);

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
    assert_grad!(df_da, vec![-4.0]);
    assert_grad!(df_db, vec![6.0]);
    assert_grad!(df_dc, vec![1.0]);

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
    assert_grad!(df_da, vec![5.0]);
    assert_grad!(df_db, vec![2.0]);
    assert_grad!(df_dc, vec![1.0]);
}

#[test]
#[serial]
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
    assert_grad!(df_da, izip!(a_vec.iter(),b_vec.iter()).map(|(a_x,b_x)|b_x*a_x.powf(b_x-1.0)).collect());
    assert_grad!(df_db, izip!(a_vec.iter(),b_vec.iter()).map(|(a_x,b_x)|a_x.powf(*b_x)*a_x.ln()).collect());
}

#[test]
#[serial]
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
    assert_grad!(dmax_da, vec![0.0, 0.5, 1.0]);
    assert_grad!(dmin_da, vec![1.0, 0.5, 0.0]);
    assert_grad!(dmax_db, vec![1.0, 0.5, 0.0]);
    assert_grad!(dmin_db, vec![0.0, 0.5, 1.0]);
}
#[test]
#[serial]
#[rustfmt::skip]
fn cmp_op() {
    let const1 = Expression::constant(3.0);
    let const2 = Expression::constant(-2.0);
    let (tensor1, _) = Expression::tensor(vec![1.0, 4.0, 3.0], true);
    let (tensor2, _) = Expression::tensor(vec![-1.0, 5.0, 3.0], true);
    let const1_eq_const2 = const1.eq(&const2);
    let const1_ne_const2 = const1.ne(&const2);
    let const1_le_const2 = const1.le(&const2);
    let const1_ge_const2 = const1.ge(&const2);
    let const1_lt_const2 = const1.lt(&const2);
    let const1_gt_const2 = const1.gt(&const2);
    assert_scalar!(&const1_eq_const2, 0.0);
    assert_scalar!(&const1_ne_const2, 1.0);
    assert_scalar!(&const1_le_const2, 0.0);
    assert_scalar!(&const1_ge_const2, 1.0);
    assert_scalar!(&const1_lt_const2, 0.0);
    assert_scalar!(&const1_gt_const2, 1.0);

    let const1_eq_tensor2 = const1.eq(&tensor2);
    let const1_ne_tensor2 = const1.ne(&tensor2);
    let const1_le_tensor2 = const1.le(&tensor2);
    let const1_ge_tensor2 = const1.ge(&tensor2);
    let const1_lt_tensor2 = const1.lt(&tensor2);
    let const1_gt_tensor2 = const1.gt(&tensor2);
    assert_tensor!(&const1_eq_tensor2, vec![0.0, 0.0, 1.0]);
    assert_tensor!(&const1_ne_tensor2, vec![1.0, 1.0, 0.0]);
    assert_tensor!(&const1_le_tensor2, vec![0.0, 1.0, 1.0]);
    assert_tensor!(&const1_ge_tensor2, vec![1.0, 0.0, 1.0]);
    assert_tensor!(&const1_lt_tensor2, vec![0.0, 1.0, 0.0]);
    assert_tensor!(&const1_gt_tensor2, vec![1.0, 0.0, 0.0]);

    let tensor1_eq_tensor2 = tensor1.eq(&tensor2);
    let tensor1_ne_tensor2 = tensor1.ne(&tensor2);
    let tensor1_le_tensor2 = tensor1.le(&tensor2);
    let tensor1_ge_tensor2 = tensor1.ge(&tensor2);
    let tensor1_lt_tensor2 = tensor1.lt(&tensor2);
    let tensor1_gt_tensor2 = tensor1.gt(&tensor2);
    assert_tensor!(&tensor1_eq_tensor2, vec![0.0, 0.0, 1.0]);
    assert_tensor!(&tensor1_ne_tensor2, vec![1.0, 1.0, 0.0]);
    assert_tensor!(&tensor1_le_tensor2, vec![0.0, 1.0, 1.0]);
    assert_tensor!(&tensor1_ge_tensor2, vec![1.0, 0.0, 1.0]);
    assert_tensor!(&tensor1_lt_tensor2, vec![0.0, 1.0, 0.0]);
    assert_tensor!(&tensor1_gt_tensor2, vec![1.0, 0.0, 0.0]);
}
#[test]
#[serial]
#[rustfmt::skip]
fn binary_op() {
    let const1 = Expression::constant(3.0);
    let const2 = Expression::constant(-2.0);
    let (tensor1, tensor1_ref) = Expression::tensor(vec![1.0, 2.0, 3.0], true);
    let (tensor2, tensor2_ref) = Expression::tensor(vec![-1.0, -2.0, -3.0], true);
    
    let const2_min_tensor2 = const2.min(&tensor2);
    let const2_max_tensor2 = const2.max(&tensor2);
    assert_tensor!(&const2_max_tensor2, vec![-1.0, -2.0, -2.0]);
    assert_tensor!(&const2_min_tensor2, vec![-2.0, -2.0, -3.0]);

    let const1_add_const2 = const1.add(&const2);
    let const1_sub_const2 = const1.sub(&const2);
    let const1_mul_const2 = const1.mul(&const2);
    let const1_div_const2 = const1.div(&const2);
    let const2_pow_const1 = const2.pow(&const1);
    assert_scalar!(&const1_add_const2,1.0);
    assert_scalar!(&const1_sub_const2,5.0);
    assert_scalar!(&const1_mul_const2,-6.0);
    assert_scalar!(&const1_div_const2,-1.5);
    assert_scalar!(&const2_pow_const1,-8.0);

    let const1_add_tensor1 = const1.add(&tensor1);
    let const1_sub_tensor1 = const1.sub(&tensor1);
    let const1_mul_tensor1 = const1.mul(&tensor1);
    let const1_div_tensor1 = const1.div(&tensor1);
    let const1_pow_tensor1 = const1.pow(&tensor1);
    assert_tensor!(&const1_add_tensor1, vec![4.0, 5.0, 6.0]);
    assert_tensor!(&const1_sub_tensor1, vec![2.0, 1.0, 0.0]);
    assert_tensor!(&const1_mul_tensor1, vec![3.0, 6.0, 9.0]);
    assert_tensor!(&const1_div_tensor1, vec![3.0, 1.5, 1.0]);
    assert_tensor!(&const1_pow_tensor1, vec![3.0, 9.0, 27.0]);

    let tensor1_add_const1 = tensor1.add(&const1);
    let tensor1_sub_const1 = tensor1.sub(&const1);
    let tensor1_mul_const1 = tensor1.mul(&const1);
    let tensor1_div_const1 = tensor1.div(&const1);
    let tensor1_pow_const1 = tensor1.pow(&const1);
    assert_tensor!(&tensor1_add_const1, vec![4.0, 5.0, 6.0]);
    assert_tensor!(&tensor1_sub_const1, vec![-2.0, -1.0, -0.0]);
    assert_tensor!(&tensor1_mul_const1, vec![3.0, 6.0, 9.0]);
    assert_tensor!(&tensor1_div_const1, vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
    assert_tensor!(&tensor1_pow_const1, vec![1.0, 8.0, 27.0]);

    let tensor1_add_tensor2 = tensor1.add(&tensor2);
    let tensor1_sub_tensor2 = tensor1.sub(&tensor2);
    let tensor1_mul_tensor2 = tensor1.mul(&tensor2);
    let tensor1_div_tensor2 = tensor1.div(&tensor2);
    let tensor2_pow_tensor1 = tensor2.pow(&tensor1);
    assert_tensor!(&tensor1_add_tensor2, vec![0.0, 0.0, 0.0]);
    assert_tensor!(&tensor1_sub_tensor2, vec![2.0, 4.0, 6.0]);
    assert_tensor!(&tensor1_mul_tensor2, vec![-1.0, -4.0, -9.0]);
    assert_tensor!(&tensor1_div_tensor2, vec![-1.0, -1.0, -1.0]);
    assert_tensor!(&tensor2_pow_tensor1, vec![-1.0, 4.0, -27.0]);

    // Compare to candle's result
    let candle_var_const1 = candle_core::Var::new(3.0, &candle_core::Device::Cpu).unwrap();
    let candle_var_const2 = candle_core::Var::new(-2.0, &candle_core::Device::Cpu).unwrap();
    let candle_var_tensor1 = candle_core::Var::new(vec![1.0, 2.0, 3.0], &candle_core::Device::Cpu).unwrap();
    let candle_var_tensor2 = candle_core::Var::new(vec![-1.0, -2.0, -3.0], &candle_core::Device::Cpu).unwrap();
    let candle_const1 = candle_var_const1.as_tensor();
    let candle_const2 = candle_var_const2.as_tensor();
    let candle_tensor1 = candle_var_tensor1.as_tensor();
    let candle_tensor2 = candle_var_tensor2.as_tensor();

    let candle_const1_add_const2 = candle_const1.add(candle_const2).unwrap();
    let candle_const1_sub_const2 = candle_const1.sub(candle_const2).unwrap();
    let candle_const1_mul_const2 = candle_const1.mul(candle_const2).unwrap();
    let candle_const1_div_const2 = candle_const1.div(candle_const2).unwrap();
    // let candle_const1_pow_const2 = candle_const1.pow(candle_const2).unwrap();
    assert_candle_scalar!(&const1_add_const2, candle_const1_add_const2);
    assert_candle_scalar!(&const1_sub_const2, candle_const1_sub_const2);
    assert_candle_scalar!(&const1_mul_const2, candle_const1_mul_const2);
    assert_candle_scalar!(&const1_div_const2, candle_const1_div_const2);
    // assert_candle_scalar!(&const2_pow_const1, candle_const2_pow_const1);

    let candle_const1_add_tensor1 = candle_const1.broadcast_add(candle_tensor1).unwrap();
    let candle_const1_sub_tensor1 = candle_const1.broadcast_sub(candle_tensor1).unwrap();
    let candle_const1_mul_tensor1 = candle_const1.broadcast_mul(candle_tensor1).unwrap();
    let candle_const1_div_tensor1 = candle_const1.broadcast_div(candle_tensor1).unwrap();
    // let candle_const1_pow_tensor1 = candle_const1.broadcast_pow(candle_tensor1).unwrap();
    assert_candle_tensor!(&const1_add_tensor1, &candle_const1_add_tensor1, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&const1_sub_tensor1, &candle_const1_sub_tensor1, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&const1_mul_tensor1, &candle_const1_mul_tensor1, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&const1_div_tensor1, &candle_const1_div_tensor1, (&tensor1_ref, candle_tensor1));

    let candle_tensor1_add_const1 = candle_tensor1.broadcast_add(candle_const1).unwrap();
    let candle_tensor1_sub_const1 = candle_tensor1.broadcast_sub(candle_const1).unwrap();
    let candle_tensor1_mul_const1 = candle_tensor1.broadcast_mul(candle_const1).unwrap();
    let candle_tensor1_div_const1 = candle_tensor1.broadcast_div(candle_const1).unwrap();
    // let candle_tensor1_pow_const1 = candle_tensor1.broadcast_pow(candle_const1).unwrap();
    assert_candle_tensor!(&tensor1_add_const1, &candle_tensor1_add_const1, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_sub_const1, &candle_tensor1_sub_const1, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_mul_const1, &candle_tensor1_mul_const1, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_div_const1, &candle_tensor1_div_const1, (&tensor1_ref, candle_tensor1));
    // assert_candle_tensor!(&tensor1_pow_const1, &candle_tensor1_pow_const1);

    let candle_tensor1_add_tensor2 = candle_tensor1.add(candle_tensor2).unwrap();
    let candle_tensor1_sub_tensor2 = candle_tensor1.sub(candle_tensor2).unwrap();
    let candle_tensor1_mul_tensor2 = candle_tensor1.mul(candle_tensor2).unwrap();
    let candle_tensor1_div_tensor2 = candle_tensor1.div(candle_tensor2).unwrap();
    // let candle_tensor2_pow_tensor1 = candle_tensor2.pow(candle_tensor1).unwrap();
    assert_candle_tensor!(&tensor1_add_tensor2, &candle_tensor1_add_tensor2, (&tensor1_ref, candle_tensor1), (&tensor2_ref, candle_tensor2));
    assert_candle_tensor!(&tensor1_sub_tensor2, &candle_tensor1_sub_tensor2, (&tensor1_ref, candle_tensor1), (&tensor2_ref, candle_tensor2));
    assert_candle_tensor!(&tensor1_mul_tensor2, &candle_tensor1_mul_tensor2, (&tensor1_ref, candle_tensor1), (&tensor2_ref, candle_tensor2));
    assert_candle_tensor!(&tensor1_div_tensor2, &candle_tensor1_div_tensor2, (&tensor1_ref, candle_tensor1), (&tensor2_ref, candle_tensor2));
    // assert_candle_tensor!(&tensor2_pow_tensor1, &candle_tensor2_pow_tensor1);

    // Update 1
    before_update();
    tensor1_ref.assgin(vec![-3.0, 6.0]);
    tensor2_ref.assgin(vec![3.0, -4.0]);

    assert_tensor!(&const2_max_tensor2, vec![3.0, -2.0]);
    assert_tensor!(&const2_min_tensor2, vec![-2.0, -4.0]);

    assert_scalar!(&const1_add_const2,1.0);
    assert_scalar!(&const1_sub_const2,5.0);
    assert_scalar!(&const1_mul_const2,-6.0);
    assert_scalar!(&const1_div_const2,-1.5);
    assert_scalar!(&const2_pow_const1,-8.0);

    assert_tensor!(&const1_add_tensor1, vec![0.0, 9.0]);
    assert_tensor!(&const1_sub_tensor1, vec![6.0, -3.0]);
    assert_tensor!(&const1_mul_tensor1, vec![-9.0, 18.0]);
    assert_tensor!(&const1_div_tensor1, vec![-1.0, 0.5]);
    assert_tensor!(&const1_pow_tensor1, vec![3.0_f64.powf(-3.0), 3.0_f64.powf(6.0)]);

    assert_tensor!(&tensor1_add_const1, vec![0.0, 9.0]);
    assert_tensor!(&tensor1_sub_const1, vec![-6.0, 3.0]);
    assert_tensor!(&tensor1_mul_const1, vec![-9.0, 18.0]);
    assert_tensor!(&tensor1_div_const1, vec![-1.0, 2.0]);
    assert_tensor!(&tensor1_pow_const1, vec![(-3.0_f64).powf(3.0), 6.0_f64.powf(3.0)]);

    assert_tensor!(&tensor1_add_tensor2, vec![0.0, 2.0]);
    assert_tensor!(&tensor1_sub_tensor2, vec![-6.0, 10.0]);
    assert_tensor!(&tensor1_mul_tensor2, vec![-9.0, -24.0]);
    assert_tensor!(&tensor1_div_tensor2, vec![-1.0, -1.5]);
    assert_tensor!(&tensor2_pow_tensor1, vec![3.0_f64.powf(-3.0), (-4.0_f64).powf(6.0)]);

    // Update 2
    before_update();
    tensor1_ref.assgin(vec![6.0]);
    tensor2_ref.assgin(vec![-4.0]);

    assert_tensor!(&const2_max_tensor2, vec![-2.0]);
    assert_tensor!(&const2_min_tensor2, vec![-4.0]);

    assert_scalar!(&const1_add_const2, 1.0);
    assert_scalar!(&const1_sub_const2, 5.0);
    assert_scalar!(&const1_mul_const2, -6.0);
    assert_scalar!(&const1_div_const2, -1.5);
    assert_scalar!(&const2_pow_const1, -8.0);

    assert_tensor!(&const1_add_tensor1, vec![9.0]);
    assert_tensor!(&const1_sub_tensor1, vec![-3.0]);
    assert_tensor!(&const1_mul_tensor1, vec![18.0]);
    assert_tensor!(&const1_div_tensor1, vec![0.5]);
    assert_tensor!(&const1_pow_tensor1, vec![3.0_f64.powf(6.0)]);

    assert_tensor!(&tensor1_add_const1, vec![9.0]);
    assert_tensor!(&tensor1_sub_const1, vec![3.0]);
    assert_tensor!(&tensor1_mul_const1, vec![18.0]);
    assert_tensor!(&tensor1_div_const1, vec![2.0]);
    assert_tensor!(&tensor1_pow_const1, vec![6.0_f64.powf(3.0)]);

    assert_tensor!(&tensor1_add_tensor2, vec![2.0]);
    assert_tensor!(&tensor1_sub_tensor2, vec![10.0]);
    assert_tensor!(&tensor1_mul_tensor2, vec![-24.0]);
    assert_tensor!(&tensor1_div_tensor2, vec![-1.5]);
    assert_tensor!(&tensor2_pow_tensor1, vec![(-4.0_f64).powf(6.0)]);
}

#[test]
#[serial]
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
    assert_tensor!(&tensor1_neg, values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sin, values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_cos, values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_tanh, values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_tan, values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_ceil, values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_floor, values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_round, values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sign, values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sqrt, values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sqr, values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_log, values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_exp, values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_abs, values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_erf, values1.iter().map(|x| candle_core::cpu::erf::erf(*x)).collect::<Vec<_>>());

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
    assert_scalar!(&const1_neg, Neg::neg(x1));
    assert_scalar!(&const1_sin, f64::sin(x1));
    assert_scalar!(&const1_cos, f64::cos(x1));
    assert_scalar!(&const1_tanh, f64::tanh(x1));
    assert_scalar!(&const1_tan, f64::tan(x1));
    assert_scalar!(&const1_ceil, f64::ceil(x1));
    assert_scalar!(&const1_floor, f64::floor(x1));
    assert_scalar!(&const1_round, f64::round(x1));
    assert_scalar!(&const1_sign, f64::signum(x1));
    assert_scalar!(&const1_sqrt, f64::sqrt(x1));
    assert_scalar!(&const1_sqr, f64::powi(x1, 2));
    assert_scalar!(&const1_log, f64::ln(x1));
    assert_scalar!(&const1_exp, f64::exp(x1));
    assert_scalar!(&const1_abs, f64::abs(x1));
    assert_scalar!(&const1_erf, candle_core::cpu::erf::erf(x1));

    let candle_var_tensor1 = candle_core::Var::new(values1.clone(), &candle_core::Device::Cpu).unwrap();
    let candle_tensor1 = candle_var_tensor1.as_tensor();

    let candle_tensor1_neg = candle_tensor1.neg().unwrap();
    let candle_tensor1_sin = candle_tensor1.sin().unwrap();
    let candle_tensor1_cos = candle_tensor1.cos().unwrap();
    let candle_tensor1_tanh = candle_tensor1.tanh().unwrap();
    let candle_tensor1_ceil = candle_tensor1.ceil().unwrap();
    let candle_tensor1_floor = candle_tensor1.floor().unwrap();
    let candle_tensor1_round = candle_tensor1.round().unwrap();
    let candle_tensor1_sign = candle_tensor1.sign().unwrap();
    let candle_tensor1_sqrt = candle_tensor1.sqrt().unwrap();
    let candle_tensor1_sqr = candle_tensor1.sqr().unwrap();
    let candle_tensor1_log = candle_tensor1.log().unwrap();
    let candle_tensor1_exp = candle_tensor1.exp().unwrap();
    let candle_tensor1_abs = candle_tensor1.abs().unwrap();
    let candle_tensor1_erf = candle_tensor1.erf().unwrap();
    assert_candle_tensor!(&tensor1_neg, &candle_tensor1_neg, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_sin, &candle_tensor1_sin, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_cos, &candle_tensor1_cos, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_tanh, &candle_tensor1_tanh, (&tensor1_ref, candle_tensor1));
    // FIXME
    assert_candle_tensor!(&tensor1_ceil, &candle_tensor1_ceil);
    // FIXME
    assert_candle_tensor!(&tensor1_floor, &candle_tensor1_floor);
    // FIXME
    assert_candle_tensor!(&tensor1_round, &candle_tensor1_round);
    // FIXME
    assert_candle_tensor!(&tensor1_sign, &candle_tensor1_sign);
    assert_candle_tensor!(&tensor1_sqrt, &candle_tensor1_sqrt, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_sqr, &candle_tensor1_sqr, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_log, &candle_tensor1_log, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_exp, &candle_tensor1_exp, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_abs, &candle_tensor1_abs, (&tensor1_ref, candle_tensor1));
    assert_candle_tensor!(&tensor1_erf, &candle_tensor1_erf, (&tensor1_ref, candle_tensor1));

    // Update1
    let values1 = vec![1.0, 2.0];
    before_update();
    tensor1_ref.assgin(values1.clone());

    assert_tensor!(&tensor1_neg, values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sin, values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_cos, values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_tanh, values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_tan, values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_ceil, values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_floor, values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_round, values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sign, values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sqrt, values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sqr, values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_log, values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_exp, values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_abs, values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_erf, values1.iter().map(|x| candle_core::cpu::erf::erf(*x)).collect::<Vec<_>>());

    assert_scalar!(&const1_neg, Neg::neg(x1));
    assert_scalar!(&const1_sin, f64::sin(x1));
    assert_scalar!(&const1_cos, f64::cos(x1));
    assert_scalar!(&const1_tanh, f64::tanh(x1));
    assert_scalar!(&const1_tan, f64::tan(x1));
    assert_scalar!(&const1_ceil, f64::ceil(x1));
    assert_scalar!(&const1_floor, f64::floor(x1));
    assert_scalar!(&const1_round, f64::round(x1));
    assert_scalar!(&const1_sign, f64::signum(x1));
    assert_scalar!(&const1_sqrt, f64::sqrt(x1));
    assert_scalar!(&const1_sqr, f64::powi(x1, 2));
    assert_scalar!(&const1_log, f64::ln(x1));
    assert_scalar!(&const1_exp, f64::exp(x1));
    assert_scalar!(&const1_abs, f64::abs(x1));
    assert_scalar!(&const1_erf, candle_core::cpu::erf::erf(x1));

    // Update2
    let values1 = vec![1.0, 2.0];
    tensor1_ref.assgin(values1.clone());
    before_update();

    assert_tensor!(&tensor1_neg, values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sin, values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_cos, values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_tanh, values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_tan, values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_ceil, values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_floor, values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_round, values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sign, values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sqrt, values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_sqr, values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_log, values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_exp, values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_abs, values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>());
    assert_tensor!(&tensor1_erf, values1.iter().map(|x| candle_core::cpu::erf::erf(*x)).collect::<Vec<_>>());

    assert_scalar!(&const1_neg, Neg::neg(x1));
    assert_scalar!(&const1_sin, f64::sin(x1));
    assert_scalar!(&const1_cos, f64::cos(x1));
    assert_scalar!(&const1_tanh, f64::tanh(x1));
    assert_scalar!(&const1_tan, f64::tan(x1));
    assert_scalar!(&const1_ceil, f64::ceil(x1));
    assert_scalar!(&const1_floor, f64::floor(x1));
    assert_scalar!(&const1_round, f64::round(x1));
    assert_scalar!(&const1_sign, f64::signum(x1));
    assert_scalar!(&const1_sqrt, f64::sqrt(x1));
    assert_scalar!(&const1_sqr, f64::powi(x1, 2));
    assert_scalar!(&const1_log, f64::ln(x1));
    assert_scalar!(&const1_exp, f64::exp(x1));
    assert_scalar!(&const1_abs, f64::abs(x1));
    assert_scalar!(&const1_erf, candle_core::cpu::erf::erf(x1));
}
