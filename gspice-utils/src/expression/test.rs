#![cfg(test)]
use itertools::izip;
use ordered_float::OrderedFloat;
use rand::prelude::Distribution;
use serial_test::serial;

use crate::{before_update, Expression};
use std::ops::*;

use super::ScalarTensor;

macro_rules! assert_eq_vec {
    ($lhs:expr, $rhs:expr) => {
        assert_eq_vec!($lhs, $rhs, 0.0);
    };
    ($lhs:expr, $rhs:expr, $tolerance:expr) => {
        let lhs = $lhs;
        let rhs = $rhs;
        assert!(
            lhs.len() == rhs.len()
                && lhs
                    .iter()
                    .zip(rhs.iter())
                    .all(|(x1, x2)| OrderedFloat(f64::abs(x1 - x2)).le(&OrderedFloat($tolerance))),
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
        assert_eq!(
            OrderedFloat($want.to_vec0::<f64>().unwrap()),
            OrderedFloat($got.value().to_scalar().unwrap())
        );
    };
}

macro_rules! assert_candle_tensor {
    ($got:expr, $want:expr) => {
        assert_eq_vec!(
            $want.to_vec1::<f64>().unwrap(),
            $got.value().to_tensor().unwrap()
        );
    };
    ($got:expr, $want:expr, ($got_tensor1:expr, $want_tensor1:expr)) => {
        assert_candle_tensor!($got, $want);
        let (grads, grads_candle) = ($got.backward(), $want.backward().unwrap());
        assert_eq_vec!(
            &grads.get($got_tensor1).unwrap(),
            &grads_candle
                .get($want_tensor1)
                .unwrap()
                .to_vec1::<f64>()
                .unwrap()
        );
    };
    ($got:expr, $want:expr, ($got_tensor1:expr, $want_tensor1:expr), ($got_tensor2:expr, $want_tensor2:expr)) => {
        assert_candle_tensor!($got, $want);
        let (grads, grads_candle) = ($got.backward(), $want.backward().unwrap());
        assert_eq_vec!(
            &grads.get($got_tensor1).unwrap(),
            &grads_candle
                .get($want_tensor1)
                .unwrap()
                .to_vec1::<f64>()
                .unwrap()
        );
        assert_eq_vec!(
            &grads.get($got_tensor2).unwrap(),
            &grads_candle
                .get($want_tensor2)
                .unwrap()
                .to_vec1::<f64>()
                .unwrap()
        );
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
    let (x, x_ref) = Expression::rand_uniform(n, -10.0, 10.0, true);
    let (y, y_ref) = Expression::rand_uniform(n, -10.0, 10.0, true);
    let f = &x.sqr() + &y.sqr();
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
fn backward_clone() {
    let (a, a_ref) = Expression::rand_uniform(10, -10.0, 10.0, true);
    let (b, b_ref) = Expression::rand_uniform(10, -10.0, 10.0, true);
    
    let f = a.mul(&b);
    let grads = f.backward();
    let df_da = grads.get(&a_ref);
    let df_db = grads.get(&b_ref);

    let clone_f = a.clone().mul(&b);
    let clone_grads = clone_f.backward();
    let clone_df_da = clone_grads.get(&a_ref);
    let clone_df_db = clone_grads.get(&b_ref);
    
    assert_eq_vec!(f.value().to_tensor().unwrap(), clone_f.value().to_tensor().unwrap());
    assert_eq_vec!(&df_da.unwrap(), &clone_df_da.unwrap());
    assert_eq_vec!(&df_db.unwrap(), &clone_df_db.unwrap());
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
fn backward_powf() {
    let a_vec = vec![1.5, 2.0, 3.0];
    let (a, a_ref) = Expression::tensor(a_vec.clone(), true);
    
    let f1 = a.powf(2.0);
    let f2 = a.powf(0.5);
    let grads1 = f1.backward();
    let grads2 = f2.backward();
    let df1_da = grads1.get(&a_ref);
    let df2_da = grads2.get(&a_ref);

    let verify_f1 = a.sqr();
    let verify_f2 = a.sqrt();
    let verify_grads1 = verify_f1.backward();
    let verify_grads2 = verify_f2.backward();
    let verify_df1_da = verify_grads1.get(&a_ref);
    let verify_df2_da = verify_grads2.get(&a_ref);

    assert_eq_vec!(f1.value().to_tensor().unwrap(), verify_f1.value().to_tensor().unwrap());
    assert_eq_vec!(f2.value().to_tensor().unwrap(), verify_f2.value().to_tensor().unwrap());
    assert_eq_vec!(&df1_da.unwrap(), &verify_df1_da.unwrap());
    assert_eq_vec!(&df2_da.unwrap(), &verify_df2_da.unwrap(), 1e-10);
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
fn backward_cond() {
    let len = 2000;
    let distr1 = rand::distributions::Bernoulli::new(0.3).unwrap();
    let distr2 = rand::distributions::Uniform::<f64>::new(-10.0, 10.0);
    let mut rng = rand::thread_rng();
    let cond_values: Vec<u8> = distr1.sample_iter(&mut rng).take(len).map(|b|if b{1}else{0}).collect();
    let a_values: Vec<f64> = distr2.sample_iter(&mut rng).take(len).collect();
    let b_values: Vec<f64> = distr2.sample_iter(&mut rng).take(len).collect();
    let (cond, cond_ref) = Expression::tensor(cond_values.iter().map(|n| *n as f64).collect(), true);
    let (a, a_ref) = Expression::tensor(a_values.clone(), true);
    let (b, b_ref) = Expression::tensor(b_values.clone(), true);
    let f = cond.cond(&a, &b);

    let candle_cond = candle_core::Tensor::new(cond_values, &candle_core::Device::Cpu).unwrap();
    let candle_a_var = candle_core::Var::new(a_values, &candle_core::Device::Cpu).unwrap();
    let candle_b_var = candle_core::Var::new(b_values, &candle_core::Device::Cpu).unwrap();
    let candle_a = candle_a_var.as_tensor();
    let candle_b = candle_b_var.as_tensor();
    let candle_f = candle_cond.where_cond(candle_a, candle_b).unwrap();
    assert_candle_tensor!(&f, &candle_f, (&a_ref, &candle_a), (&b_ref, &candle_b));

    before_update();
    let cond_values: Vec<u8> = distr1.sample_iter(&mut rng).take(len).map(|b|if b{1}else{0}).collect();
    let a_values: Vec<f64> = distr2.sample_iter(&mut rng).take(len).collect();
    let b_values: Vec<f64> = distr2.sample_iter(&mut rng).take(len).collect();
    cond_ref.assgin(cond_values.iter().map(|n| *n as f64).collect());
    a_ref.assgin(a_values.clone());
    b_ref.assgin(b_values.clone());
    let candle_cond = candle_core::Tensor::new(cond_values, &candle_core::Device::Cpu).unwrap();
    let candle_a_var = candle_core::Var::new(a_values, &candle_core::Device::Cpu).unwrap();
    let candle_b_var = candle_core::Var::new(b_values, &candle_core::Device::Cpu).unwrap();
    let candle_a = candle_a_var.as_tensor();
    let candle_b = candle_b_var.as_tensor();
    let candle_f = candle_cond.where_cond(candle_a, candle_b).unwrap();
    assert_candle_tensor!(&f, &candle_f, (&a_ref, &candle_a), (&b_ref, &candle_b));

    before_update();
    let cond_values: Vec<u8> = distr1.sample_iter(&mut rng).take(len).map(|b|if b{1}else{0}).collect();
    let a_values: Vec<f64> = distr2.sample_iter(&mut rng).take(len).collect();
    let b_values: Vec<f64> = distr2.sample_iter(&mut rng).take(len).collect();
    cond_ref.assgin(cond_values.iter().map(|n| *n as f64).collect());
    a_ref.assgin(a_values.clone());
    b_ref.assgin(b_values.clone());
    let candle_cond = candle_core::Tensor::new(cond_values, &candle_core::Device::Cpu).unwrap();
    let candle_a_var = candle_core::Var::new(a_values, &candle_core::Device::Cpu).unwrap();
    let candle_b_var = candle_core::Var::new(b_values, &candle_core::Device::Cpu).unwrap();
    let candle_a = candle_a_var.as_tensor();
    let candle_b = candle_b_var.as_tensor();
    let candle_f = candle_cond.where_cond(candle_a, candle_b).unwrap();
    assert_candle_tensor!(&f, &candle_f, (&a_ref, &candle_a), (&b_ref, &candle_b));
}

#[test]
#[serial]
#[rustfmt::skip]
fn backward_cond_logic() {
    let epsilon = 0.3;
    let k = 10.0;
    let (a1, a1_ref) = Expression::tensor(vec![0.9, 1.1, 1.0], true);
    let (b1, b1_ref) = Expression::tensor(vec![1.0, 1.0, 1.1], true);
    let (a2, a2_ref) = Expression::tensor(vec![0.9, 1.15, 0.7], true);
    let (b2, b2_ref) = Expression::tensor(vec![1.1, 1.0, 1.1], true);
    let (c, c_ref) = Expression::tensor(vec![10., 10., 10.], true);
    let (d, d_ref) = Expression::tensor(vec![0.0, 0.0, 0.0], true);

    println!("\nlinear Equal\n");
    let linear_a1_eq_b1 = a1.eq_linear(&b1, epsilon);
    let linear_a2_eq_b2 = a2.eq_linear(&b2, epsilon);
    let linear_a1_eq_b1_cond_c_d = linear_a1_eq_b1.cond(&c, &d);
    let linear_a2_eq_b2_cond_c_d = linear_a2_eq_b2.cond(&c, &d);
    let linear_a1_eq_b1_cond_c_d_grads = linear_a1_eq_b1_cond_c_d.backward();
    let linear_a2_eq_b2_cond_c_d_grads = linear_a2_eq_b2_cond_c_d.backward();
    let linear_a1_eq_b1_cond_c_d_grad_da1 = linear_a1_eq_b1_cond_c_d_grads.get(&a1_ref).unwrap();
    let linear_a2_eq_b2_cond_c_d_grad_da2 = linear_a2_eq_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let linear_a1_eq_b1_cond_c_d_grad_db1 = linear_a1_eq_b1_cond_c_d_grads.get(&b1_ref).unwrap();
    let linear_a2_eq_b2_cond_c_d_grad_db2 = linear_a2_eq_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let linear_a1_eq_b1_cond_c_d_grad_dc = linear_a1_eq_b1_cond_c_d_grads.get(&c_ref).unwrap();
    let linear_a2_eq_b2_cond_c_d_grad_dc = linear_a2_eq_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let linear_a1_eq_b1_cond_c_d_grad_dd = linear_a1_eq_b1_cond_c_d_grads.get(&d_ref).unwrap();
    let linear_a2_eq_b2_cond_c_d_grad_dd = linear_a2_eq_b2_cond_c_d_grads.get(&d_ref).unwrap();
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d = linear_a1_eq_b1.logic_or(&linear_a2_eq_b2).cond(&c, &d);
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads = linear_a1_eq_b1_or_a2_eq_b2_cond_c_d.backward();
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a1 = linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&a1_ref).unwrap();
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a2 = linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b1 = linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&b1_ref).unwrap();
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b2 = linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_c = linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_d = linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&d_ref).unwrap();
    println!("f1 = (a1==b1)? c : d  {linear_a1_eq_b1_cond_c_d}");
    println!("f2 = (a2==b2)? c : d  {linear_a2_eq_b2_cond_c_d}");
    println!("∂f1/∂a1  {linear_a1_eq_b1_cond_c_d_grad_da1}");
    println!("∂f2/∂a2  {linear_a2_eq_b2_cond_c_d_grad_da2}");
    println!("∂f1/∂b1  {linear_a1_eq_b1_cond_c_d_grad_db1}");
    println!("∂f2/∂b2  {linear_a2_eq_b2_cond_c_d_grad_db2}");
    println!("∂f1/∂c  {linear_a1_eq_b1_cond_c_d_grad_dc}");
    println!("∂f2/∂c  {linear_a2_eq_b2_cond_c_d_grad_dc}");
    println!("∂f1/∂d  {linear_a1_eq_b1_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {linear_a2_eq_b2_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {linear_a2_eq_b2_cond_c_d_grad_dd}");
    println!("f3 = (a1==b1|a2==b2)? c : d  {linear_a1_eq_b1_or_a2_eq_b2_cond_c_d}");
    println!("∂f3/∂a1  {linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a1}");
    println!("∂f3/∂a2  {linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a2}");
    println!("∂f3/∂b1  {linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b1}");
    println!("∂f3/∂b2  {linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b2}");
    println!("∂f3/∂c  {linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_c}");
    println!("∂f3/∂d  {linear_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_d}");
    
    println!("\nlinear Less Than\n");
    let linear_a1_lt_b1 = a1.lt_linear(&b1, epsilon);
    let linear_a2_lt_b2 = a2.lt_linear(&b2, epsilon);
    let linear_a1_lt_b1_cond_c_d = linear_a1_lt_b1.cond(&c, &d);
    let linear_a2_lt_b2_cond_c_d = linear_a2_lt_b2.cond(&c, &d);
    let linear_a1_lt_b1_cond_c_d_grads = linear_a1_lt_b1_cond_c_d.backward();
    let linear_a2_lt_b2_cond_c_d_grads = linear_a2_lt_b2_cond_c_d.backward();
    let linear_a1_lt_b1_cond_c_d_grad_da1 = linear_a1_lt_b1_cond_c_d_grads.get(&a1_ref).unwrap();
    let linear_a2_lt_b2_cond_c_d_grad_da2 = linear_a2_lt_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let linear_a1_lt_b1_cond_c_d_grad_db1 = linear_a1_lt_b1_cond_c_d_grads.get(&b1_ref).unwrap();
    let linear_a2_lt_b2_cond_c_d_grad_db2 = linear_a2_lt_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let linear_a1_lt_b1_cond_c_d_grad_dc = linear_a1_lt_b1_cond_c_d_grads.get(&c_ref).unwrap();
    let linear_a2_lt_b2_cond_c_d_grad_dc = linear_a2_lt_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let linear_a1_lt_b1_cond_c_d_grad_dd = linear_a1_lt_b1_cond_c_d_grads.get(&d_ref).unwrap();
    let linear_a2_lt_b2_cond_c_d_grad_dd = linear_a2_lt_b2_cond_c_d_grads.get(&d_ref).unwrap();
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d = linear_a1_lt_b1.logic_or(&linear_a2_lt_b2).cond(&c, &d);
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads = linear_a1_lt_b1_or_a2_lt_b2_cond_c_d.backward();
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a1 = linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&a1_ref).unwrap();
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a2 = linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b1 = linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&b1_ref).unwrap();
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b2 = linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_c = linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_d = linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&d_ref).unwrap();
    println!("f1 = (a1<b1)? c : d  {linear_a1_lt_b1_cond_c_d}");
    println!("f2 = (a2<b2)? c : d  {linear_a2_lt_b2_cond_c_d}");
    println!("∂f1/∂a1  {linear_a1_lt_b1_cond_c_d_grad_da1}");
    println!("∂f2/∂a2  {linear_a2_lt_b2_cond_c_d_grad_da2}");
    println!("∂f1/∂b1  {linear_a1_lt_b1_cond_c_d_grad_db1}");
    println!("∂f2/∂b2  {linear_a2_lt_b2_cond_c_d_grad_db2}");
    println!("∂f1/∂c  {linear_a1_lt_b1_cond_c_d_grad_dc}");
    println!("∂f2/∂c  {linear_a2_lt_b2_cond_c_d_grad_dc}");
    println!("∂f1/∂d  {linear_a1_lt_b1_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {linear_a2_lt_b2_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {linear_a2_lt_b2_cond_c_d_grad_dd}");
    println!("f3 = (a1<b1|a2<b2)? c : d  {linear_a1_lt_b1_or_a2_lt_b2_cond_c_d}");
    println!("∂f3/∂a1  {linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a1}");
    println!("∂f3/∂a2  {linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a2}");
    println!("∂f3/∂b1  {linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b1}");
    println!("∂f3/∂b2  {linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b2}");
    println!("∂f3/∂c  {linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_c}");
    println!("∂f3/∂d  {linear_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_d}");

    println!("\nsigmoid Equal\n");
    let sigmoid_a1_eq_b1 = a1.eq_sigmoid(&b1, k);
    let sigmoid_a2_eq_b2 = a2.eq_sigmoid(&b2, k);
    let sigmoid_a1_eq_b1_cond_c_d = sigmoid_a1_eq_b1.cond(&c, &d);
    let sigmoid_a2_eq_b2_cond_c_d = sigmoid_a2_eq_b2.cond(&c, &d);
    let sigmoid_a1_eq_b1_cond_c_d_grads = sigmoid_a1_eq_b1_cond_c_d.backward();
    let sigmoid_a2_eq_b2_cond_c_d_grads = sigmoid_a2_eq_b2_cond_c_d.backward();
    let sigmoid_a1_eq_b1_cond_c_d_grad_da1 = sigmoid_a1_eq_b1_cond_c_d_grads.get(&a1_ref).unwrap();
    let sigmoid_a2_eq_b2_cond_c_d_grad_da2 = sigmoid_a2_eq_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let sigmoid_a1_eq_b1_cond_c_d_grad_db1 = sigmoid_a1_eq_b1_cond_c_d_grads.get(&b1_ref).unwrap();
    let sigmoid_a2_eq_b2_cond_c_d_grad_db2 = sigmoid_a2_eq_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let sigmoid_a1_eq_b1_cond_c_d_grad_dc = sigmoid_a1_eq_b1_cond_c_d_grads.get(&c_ref).unwrap();
    let sigmoid_a2_eq_b2_cond_c_d_grad_dc = sigmoid_a2_eq_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let sigmoid_a1_eq_b1_cond_c_d_grad_dd = sigmoid_a1_eq_b1_cond_c_d_grads.get(&d_ref).unwrap();
    let sigmoid_a2_eq_b2_cond_c_d_grad_dd = sigmoid_a2_eq_b2_cond_c_d_grads.get(&d_ref).unwrap();
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d = sigmoid_a1_eq_b1.logic_or(&sigmoid_a2_eq_b2).cond(&c, &d);
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads = sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d.backward();
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a1 = sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&a1_ref).unwrap();
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a2 = sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b1 = sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&b1_ref).unwrap();
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b2 = sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_c = sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_d = sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grads.get(&d_ref).unwrap();
    println!("f1 = (a1==b1)? c : d  {sigmoid_a1_eq_b1_cond_c_d}");
    println!("f2 = (a2==b2)? c : d  {sigmoid_a2_eq_b2_cond_c_d}");
    println!("∂f1/∂a1  {sigmoid_a1_eq_b1_cond_c_d_grad_da1}");
    println!("∂f2/∂a2  {sigmoid_a2_eq_b2_cond_c_d_grad_da2}");
    println!("∂f1/∂b1  {sigmoid_a1_eq_b1_cond_c_d_grad_db1}");
    println!("∂f2/∂b2  {sigmoid_a2_eq_b2_cond_c_d_grad_db2}");
    println!("∂f1/∂c  {sigmoid_a1_eq_b1_cond_c_d_grad_dc}");
    println!("∂f2/∂c  {sigmoid_a2_eq_b2_cond_c_d_grad_dc}");
    println!("∂f1/∂d  {sigmoid_a1_eq_b1_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {sigmoid_a2_eq_b2_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {sigmoid_a2_eq_b2_cond_c_d_grad_dd}");
    println!("f3 = (a1==b1|a2==b2)? c : d  {sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d}");
    println!("∂f3/∂a1  {sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a1}");
    println!("∂f3/∂a2  {sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_a2}");
    println!("∂f3/∂b1  {sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b1}");
    println!("∂f3/∂b2  {sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_b2}");
    println!("∂f3/∂c  {sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_c}");
    println!("∂f3/∂d  {sigmoid_a1_eq_b1_or_a2_eq_b2_cond_c_d_grad_d}");
    
    println!("\nsigmoid Less Than\n");
    let sigmoid_a1_lt_b1 = a1.lt_sigmoid(&b1, epsilon);
    let sigmoid_a2_lt_b2 = a2.lt_sigmoid(&b2, epsilon);
    let sigmoid_a1_lt_b1_cond_c_d = sigmoid_a1_lt_b1.cond(&c, &d);
    let sigmoid_a2_lt_b2_cond_c_d = sigmoid_a2_lt_b2.cond(&c, &d);
    let sigmoid_a1_lt_b1_cond_c_d_grads = sigmoid_a1_lt_b1_cond_c_d.backward();
    let sigmoid_a2_lt_b2_cond_c_d_grads = sigmoid_a2_lt_b2_cond_c_d.backward();
    let sigmoid_a1_lt_b1_cond_c_d_grad_da1 = sigmoid_a1_lt_b1_cond_c_d_grads.get(&a1_ref).unwrap();
    let sigmoid_a2_lt_b2_cond_c_d_grad_da2 = sigmoid_a2_lt_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let sigmoid_a1_lt_b1_cond_c_d_grad_db1 = sigmoid_a1_lt_b1_cond_c_d_grads.get(&b1_ref).unwrap();
    let sigmoid_a2_lt_b2_cond_c_d_grad_db2 = sigmoid_a2_lt_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let sigmoid_a1_lt_b1_cond_c_d_grad_dc = sigmoid_a1_lt_b1_cond_c_d_grads.get(&c_ref).unwrap();
    let sigmoid_a2_lt_b2_cond_c_d_grad_dc = sigmoid_a2_lt_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let sigmoid_a1_lt_b1_cond_c_d_grad_dd = sigmoid_a1_lt_b1_cond_c_d_grads.get(&d_ref).unwrap();
    let sigmoid_a2_lt_b2_cond_c_d_grad_dd = sigmoid_a2_lt_b2_cond_c_d_grads.get(&d_ref).unwrap();
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d = sigmoid_a1_lt_b1.logic_or(&sigmoid_a2_lt_b2).cond(&c, &d);
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads = sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d.backward();
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a1 = sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&a1_ref).unwrap();
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a2 = sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&a2_ref).unwrap();
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b1 = sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&b1_ref).unwrap();
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b2 = sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&b2_ref).unwrap();
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_c = sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&c_ref).unwrap();
    let sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_d = sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grads.get(&d_ref).unwrap();
    println!("f1 = (a1<b1)? c : d  {sigmoid_a1_lt_b1_cond_c_d}");
    println!("f2 = (a2<b2)? c : d  {sigmoid_a2_lt_b2_cond_c_d}");
    println!("∂f1/∂a1  {sigmoid_a1_lt_b1_cond_c_d_grad_da1}");
    println!("∂f2/∂a2  {sigmoid_a2_lt_b2_cond_c_d_grad_da2}");
    println!("∂f1/∂b1  {sigmoid_a1_lt_b1_cond_c_d_grad_db1}");
    println!("∂f2/∂b2  {sigmoid_a2_lt_b2_cond_c_d_grad_db2}");
    println!("∂f1/∂c  {sigmoid_a1_lt_b1_cond_c_d_grad_dc}");
    println!("∂f2/∂c  {sigmoid_a2_lt_b2_cond_c_d_grad_dc}");
    println!("∂f1/∂d  {sigmoid_a1_lt_b1_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {sigmoid_a2_lt_b2_cond_c_d_grad_dd}");
    println!("∂f2/∂d  {sigmoid_a2_lt_b2_cond_c_d_grad_dd}");
    println!("f3 = (a1<b1|a2<b2)? c : d  {sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d}");
    println!("∂f3/∂a1  {sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a1}");
    println!("∂f3/∂a2  {sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_a2}");
    println!("∂f3/∂b1  {sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b1}");
    println!("∂f3/∂b2  {sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_b2}");
    println!("∂f3/∂c  {sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_c}");
    println!("∂f3/∂d  {sigmoid_a1_lt_b1_or_a2_lt_b2_cond_c_d_grad_d}");
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
