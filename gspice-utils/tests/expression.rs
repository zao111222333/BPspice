#![cfg(test)]
use gspice_utils::{Expression, before_update};
use std::ops::*;

#[test]
fn op() {
    // can NOT run test parallelly,
    // since the test functions will use global COUNTER
    binary_op();
    unary_op();
}

fn binary_op() {
    let const1 = Expression::constant(3.0);
    let const2 = Expression::constant(-2.0);
    let (param1, param1_tensor) = Expression::tensor(vec![1.0, 2.0, 3.0], true);
    let (param2, param2_tensor) = Expression::tensor(vec![-1.0, -2.0, -3.0], true);

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

    assert!(const2_max_param2.value().eq_vec(&vec![-1.0, -2.0, -2.0]));
    assert!(const2_min_param2.value().eq_vec(&vec![-2.0, -2.0, -3.0]));

    assert!(const1_add_const2.value().eq_num(1.0));
    assert!(const1_sub_const2.value().eq_num(5.0));
    assert!(const1_mul_const2.value().eq_num(-6.0));
    assert!(const1_div_const2.value().eq_num(-1.5));
    assert!(const2_pow_const1.value().eq_num(-8.0));

    assert!(const1_add_param1.value().eq_vec(&vec![4.0, 5.0, 6.0]));
    assert!(const1_sub_param1.value().eq_vec(&vec![2.0, 1.0, 0.0]));
    assert!(const1_mul_param1.value().eq_vec(&vec![3.0, 6.0, 9.0]));
    assert!(const1_div_param1.value().eq_vec(&vec![3.0, 1.5, 1.0]));
    assert!(const1_pow_param1.value().eq_vec(&vec![3.0, 9.0, 27.0]));

    assert!(param1_add_const1.value().eq_vec(&vec![4.0, 5.0, 6.0]));
    assert!(param1_sub_const1.value().eq_vec(&vec![-2.0, -1.0, -0.0]));
    assert!(param1_mul_const1.value().eq_vec(&vec![3.0, 6.0, 9.0]));
    assert!(param1_div_const1.value().eq_vec(&vec![1.0 / 3.0, 2.0 / 3.0, 1.0]));
    assert!(param1_pow_const1.value().eq_vec(&vec![1.0, 8.0, 27.0]));

    assert!(param1_add_param2.value().eq_vec(&vec![0.0, 0.0, 0.0]));
    assert!(param1_sub_param2.value().eq_vec(&vec![2.0, 4.0, 6.0]));
    assert!(param1_mul_param2.value().eq_vec(&vec![-1.0, -4.0, -9.0]));
    assert!(param1_div_param2.value().eq_vec(&vec![-1.0, -1.0, -1.0]));
    assert!(param2_pow_param1.value().eq_vec(&vec![-1.0, 4.0, -27.0]));

    // Update 1
    before_update();
    param1_tensor.update(vec![-3.0, 6.0]);
    param2_tensor.update(vec![3.0, -4.0]);

    assert!(const2_max_param2.value().eq_vec(&vec![3.0, -2.0]));
    assert!(const2_min_param2.value().eq_vec(&vec![-2.0, -4.0]));

    assert!(const1_add_const2.value().eq_num(1.0));
    assert!(const1_sub_const2.value().eq_num(5.0));
    assert!(const1_mul_const2.value().eq_num(-6.0));
    assert!(const1_div_const2.value().eq_num(-1.5));
    assert!(const2_pow_const1.value().eq_num(-8.0));

    assert!(const1_add_param1.value().eq_vec(&vec![0.0, 9.0]));
    assert!(const1_sub_param1.value().eq_vec(&vec![6.0, -3.0]));
    assert!(const1_mul_param1.value().eq_vec(&vec![-9.0, 18.0]));
    assert!(const1_div_param1.value().eq_vec(&vec![-1.0, 0.5]));
    assert!(const1_pow_param1.value().eq_vec(&vec![3.0_f64.powf(-3.0), 3.0_f64.powf(6.0)]));

    assert!(param1_add_const1.value().eq_vec(&vec![0.0, 9.0]));
    assert!(param1_sub_const1.value().eq_vec(&vec![-6.0, 3.0]));
    assert!(param1_mul_const1.value().eq_vec(&vec![-9.0, 18.0]));
    assert!(param1_div_const1.value().eq_vec(&vec![-1.0, 2.0]));
    assert!(param1_pow_const1.value().eq_vec(&vec![(-3.0_f64).powf(3.0), 6.0_f64.powf(3.0)]));

    assert!(param1_add_param2.value().eq_vec(&vec![0.0, 2.0]));
    assert!(param1_sub_param2.value().eq_vec(&vec![-6.0, 10.0]));
    assert!(param1_mul_param2.value().eq_vec(&vec![-9.0, -24.0]));
    assert!(param1_div_param2.value().eq_vec(&vec![-1.0, -1.5]));
    assert!(param2_pow_param1.value().eq_vec(&vec![3.0_f64.powf(-3.0), (-4.0_f64).powf(6.0)]));

    // Update 2
    before_update();
    param1_tensor.update(vec![6.0]);
    param2_tensor.update(vec![-4.0]);

    assert!(const2_max_param2.value().eq_vec(&vec![-2.0]));
    assert!(const2_min_param2.value().eq_vec(&vec![-4.0]));

    assert!(const1_add_const2.value().eq_num(1.0));
    assert!(const1_sub_const2.value().eq_num(5.0));
    assert!(const1_mul_const2.value().eq_num(-6.0));
    assert!(const1_div_const2.value().eq_num(-1.5));
    assert!(const2_pow_const1.value().eq_num(-8.0));

    assert!(const1_add_param1.value().eq_vec(&vec![9.0]));
    assert!(const1_sub_param1.value().eq_vec(&vec![-3.0]));
    assert!(const1_mul_param1.value().eq_vec(&vec![18.0]));
    assert!(const1_div_param1.value().eq_vec(&vec![0.5]));
    assert!(const1_pow_param1.value().eq_vec(&vec![3.0_f64.powf(6.0)]));

    assert!(param1_add_const1.value().eq_vec(&vec![9.0]));
    assert!(param1_sub_const1.value().eq_vec(&vec![3.0]));
    assert!(param1_mul_const1.value().eq_vec(&vec![18.0]));
    assert!(param1_div_const1.value().eq_vec(&vec![2.0]));
    assert!(param1_pow_const1.value().eq_vec(&vec![6.0_f64.powf(3.0)]));

    assert!(param1_add_param2.value().eq_vec(&vec![2.0]));
    assert!(param1_sub_param2.value().eq_vec(&vec![10.0]));
    assert!(param1_mul_param2.value().eq_vec(&vec![-24.0]));
    assert!(param1_div_param2.value().eq_vec(&vec![-1.5]));
    assert!(param2_pow_param1.value().eq_vec(&vec![(-4.0_f64).powf(6.0)]));
}

fn unary_op() {
    let values1 = vec![1.0, 2.0, 3.0];
    let x1 = 3.0;
    let const1 = Expression::constant(x1);
    let (param1, param1_tensor) = Expression::tensor(values1.clone(), true);

    let param1_neg = param1.neg();
    let param1_sin = param1.sin();
    let param1_cos = param1.cos();
    let param1_tanh = param1.tanh();
    let param1_tan = param1.tan();
    let param1_ceil = param1.ceil();
    let param1_floor = param1.floor();
    let param1_round = param1.round();
    let param1_sign = param1.sign();
    let param1_sqrt = param1.sqrt();
    let param1_sqr = param1.sqr();
    let param1_log = param1.log();
    let param1_exp = param1.exp();
    let param1_abs = param1.abs();
    let param1_erf = param1.erf();

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

    assert!(param1_neg.value().eq_vec(&values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>()));
    assert!(param1_sin.value().eq_vec(&values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>()));
    assert!(param1_cos.value().eq_vec(&values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>()));
    assert!(param1_tanh.value().eq_vec(&values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>()));
    assert!(param1_tan.value().eq_vec(&values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>()));
    assert!(param1_ceil.value().eq_vec(&values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>()));
    assert!(param1_floor.value().eq_vec(&values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>()));
    assert!(param1_round.value().eq_vec(&values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>()));
    assert!(param1_sign.value().eq_vec(&values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>()));
    assert!(param1_sqrt.value().eq_vec(&values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>()));
    assert!(param1_sqr.value().eq_vec(&values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>()));
    assert!(param1_log.value().eq_vec(&values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>()));
    assert!(param1_exp.value().eq_vec(&values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>()));
    assert!(param1_abs.value().eq_vec(&values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>()));
    assert!(param1_erf.value().eq_vec(
        &values1
            .iter()
            .map(|x| candle_core::cpu::erf::erf(*x))
            .collect::<Vec<_>>()
    ));

    assert!(const1_neg.value().eq_num(Neg::neg(x1)));
    assert!(const1_sin.value().eq_num(f64::sin(x1)));
    assert!(const1_cos.value().eq_num(f64::cos(x1)));
    assert!(const1_tanh.value().eq_num(f64::tanh(x1)));
    assert!(const1_tan.value().eq_num(f64::tan(x1)));
    assert!(const1_ceil.value().eq_num(f64::ceil(x1)));
    assert!(const1_floor.value().eq_num(f64::floor(x1)));
    assert!(const1_round.value().eq_num(f64::round(x1)));
    assert!(const1_sign.value().eq_num(f64::signum(x1)));
    assert!(const1_sqrt.value().eq_num(f64::sqrt(x1)));
    assert!(const1_sqr.value().eq_num(f64::powi(x1, 2)));
    assert!(const1_log.value().eq_num(f64::ln(x1)));
    assert!(const1_exp.value().eq_num(f64::exp(x1)));
    assert!(const1_abs.value().eq_num(f64::abs(x1)));
    assert!(const1_erf.value().eq_num(candle_core::cpu::erf::erf(x1)));

    // Update1
    let values1 = vec![1.0, 2.0];
    before_update();
    param1_tensor.update(values1.clone());

    assert!(param1_neg.value().eq_vec(&values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>()));
    assert!(param1_sin.value().eq_vec(&values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>()));
    assert!(param1_cos.value().eq_vec(&values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>()));
    assert!(param1_tanh.value().eq_vec(&values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>()));
    assert!(param1_tan.value().eq_vec(&values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>()));
    assert!(param1_ceil.value().eq_vec(&values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>()));
    assert!(param1_floor.value().eq_vec(&values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>()));
    assert!(param1_round.value().eq_vec(&values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>()));
    assert!(param1_sign.value().eq_vec(&values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>()));
    assert!(param1_sqrt.value().eq_vec(&values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>()));
    assert!(param1_sqr.value().eq_vec(&values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>()));
    assert!(param1_log.value().eq_vec(&values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>()));
    assert!(param1_exp.value().eq_vec(&values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>()));
    assert!(param1_abs.value().eq_vec(&values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>()));
    assert!(param1_erf.value().eq_vec(
        &values1
            .iter()
            .map(|x| candle_core::cpu::erf::erf(*x))
            .collect::<Vec<_>>()
    ));

    assert!(const1_neg.value().eq_num(Neg::neg(x1)));
    assert!(const1_sin.value().eq_num(f64::sin(x1)));
    assert!(const1_cos.value().eq_num(f64::cos(x1)));
    assert!(const1_tanh.value().eq_num(f64::tanh(x1)));
    assert!(const1_tan.value().eq_num(f64::tan(x1)));
    assert!(const1_ceil.value().eq_num(f64::ceil(x1)));
    assert!(const1_floor.value().eq_num(f64::floor(x1)));
    assert!(const1_round.value().eq_num(f64::round(x1)));
    assert!(const1_sign.value().eq_num(f64::signum(x1)));
    assert!(const1_sqrt.value().eq_num(f64::sqrt(x1)));
    assert!(const1_sqr.value().eq_num(f64::powi(x1, 2)));
    assert!(const1_log.value().eq_num(f64::ln(x1)));
    assert!(const1_exp.value().eq_num(f64::exp(x1)));
    assert!(const1_abs.value().eq_num(f64::abs(x1)));
    assert!(const1_erf.value().eq_num(candle_core::cpu::erf::erf(x1)));

    // Update2
    let values1 = vec![1.0, 2.0];
    param1_tensor.update(values1.clone());
    before_update();

    assert!(param1_neg.value().eq_vec(&values1.iter().map(|x| Neg::neg(x)).collect::<Vec<_>>()));
    assert!(param1_sin.value().eq_vec(&values1.iter().map(|x| f64::sin(*x)).collect::<Vec<_>>()));
    assert!(param1_cos.value().eq_vec(&values1.iter().map(|x| f64::cos(*x)).collect::<Vec<_>>()));
    assert!(param1_tanh.value().eq_vec(&values1.iter().map(|x| f64::tanh(*x)).collect::<Vec<_>>()));
    assert!(param1_tan.value().eq_vec(&values1.iter().map(|x| f64::tan(*x)).collect::<Vec<_>>()));
    assert!(param1_ceil.value().eq_vec(&values1.iter().map(|x| f64::ceil(*x)).collect::<Vec<_>>()));
    assert!(param1_floor.value().eq_vec(&values1.iter().map(|x| f64::floor(*x)).collect::<Vec<_>>()));
    assert!(param1_round.value().eq_vec(&values1.iter().map(|x| f64::round(*x)).collect::<Vec<_>>()));
    assert!(param1_sign.value().eq_vec(&values1.iter().map(|x| f64::signum(*x)).collect::<Vec<_>>()));
    assert!(param1_sqrt.value().eq_vec(&values1.iter().map(|x| f64::sqrt(*x)).collect::<Vec<_>>()));
    assert!(param1_sqr.value().eq_vec(&values1.iter().map(|x| f64::powi(*x, 2)).collect::<Vec<_>>()));
    assert!(param1_log.value().eq_vec(&values1.iter().map(|x| f64::ln(*x)).collect::<Vec<_>>()));
    assert!(param1_exp.value().eq_vec(&values1.iter().map(|x| f64::exp(*x)).collect::<Vec<_>>()));
    assert!(param1_abs.value().eq_vec(&values1.iter().map(|x| f64::abs(*x)).collect::<Vec<_>>()));
    assert!(param1_erf.value().eq_vec(
        &values1
            .iter()
            .map(|x| candle_core::cpu::erf::erf(*x))
            .collect::<Vec<_>>()
    ));

    assert!(const1_neg.value().eq_num(Neg::neg(x1)));
    assert!(const1_sin.value().eq_num(f64::sin(x1)));
    assert!(const1_cos.value().eq_num(f64::cos(x1)));
    assert!(const1_tanh.value().eq_num(f64::tanh(x1)));
    assert!(const1_tan.value().eq_num(f64::tan(x1)));
    assert!(const1_ceil.value().eq_num(f64::ceil(x1)));
    assert!(const1_floor.value().eq_num(f64::floor(x1)));
    assert!(const1_round.value().eq_num(f64::round(x1)));
    assert!(const1_sign.value().eq_num(f64::signum(x1)));
    assert!(const1_sqrt.value().eq_num(f64::sqrt(x1)));
    assert!(const1_sqr.value().eq_num(f64::powi(x1, 2)));
    assert!(const1_log.value().eq_num(f64::ln(x1)));
    assert!(const1_exp.value().eq_num(f64::exp(x1)));
    assert!(const1_abs.value().eq_num(f64::abs(x1)));
    assert!(const1_erf.value().eq_num(candle_core::cpu::erf::erf(x1)));
}
