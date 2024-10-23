use ordered_float::OrderedFloat;

use super::{
    autograd::Grad,
    op::{
        DiscreteBinaryOp, DiscreteBinaryOpT, GradMethod, GradMethodDiscrete, GradMethodLinear, GradMethodSigmoid,
        GradMethodT,
    },
    Expression, ScalarTensor, Tensor,
};
use core::fmt::{self, Write};

pub(crate) fn fmt_vec(vec: &[f64], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut buffer = ryu::Buffer::new();
    let len = vec.len();
    if len >= 100 {
        write!(f, "([{}, ", buffer.format(vec[0]))?;
        write!(f, "{}, ", buffer.format(vec[1]))?;
        write!(f, "{}, ..., ", buffer.format(vec[2]))?;
        write!(f, "{}, ", buffer.format(vec[len - 3]))?;
        write!(f, "{}, ", buffer.format(vec[len - 2]))?;
        write!(f, "{}]) [{len}x1]", buffer.format(vec[len - 1]))
    } else {
        let mut iter = vec.iter();
        if let Some(first) = iter.next() {
            f.write_char('[')?;
            f.write_str(buffer.format(*first))?;
            for x in iter {
                write!(f, ", {}", buffer.format(*x))?;
            }
        }
        write!(f, "])")
    }
}
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_vec(&self.values().read().unwrap(), f)
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Const(v) => write!(f, "Const({})", v),
            Expression::Tensor(tensor) => {
                write!(f, "Tensor{}", tensor)
            }
        }
    }
}
impl fmt::Display for Grad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Grad")?;
        fmt_vec(&self.0, f)
    }
}

impl<'a> ScalarTensor<'a> {
    pub fn to_scalar(&self) -> Option<f64> {
        if let ScalarTensor::Scalar(f) = self {
            Some(**f)
        } else {
            None
        }
    }
    pub fn to_tensor(&self) -> Option<Vec<f64>> {
        if let ScalarTensor::Tensor(tensor) = self {
            Some(tensor.read().unwrap().clone())
        } else {
            None
        }
    }
    // #[cfg(test)]
    pub fn overall_sum(&self) -> f64 {
        match self {
            ScalarTensor::Scalar(x) => **x,
            ScalarTensor::Tensor(tensor) => {
                tensor.read().unwrap().iter().fold(0.0, |sum, x| sum + x)
            }
        }
    }
}
impl<'a> fmt::Display for ScalarTensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarTensor::Scalar(v) => write!(f, "Scalar({})", v),
            ScalarTensor::Tensor(tensor) => {
                write!(f, "Tensor")?;
                fmt_vec(&tensor.read().unwrap(), f)
            }
        }
    }
}

pub(super) trait DiscreteBinaryIter: GradMethodT {
    #[inline]
    fn eq_backward_lhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, lhs_sum_grad)| {
            self.eq_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn eq_backward_rhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, rhs_sum_grad)| {
            self.eq_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad)
        });
    }
    #[inline]
    fn eq_backward_lhs_iter_fix_rhs<'a>(
        &self,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        lhs_iter.for_each(|(lhs, res, grad, lhs_sum_grad)| {
            self.eq_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn eq_backward_rhs_iter_fix_lhs<'a>(
        &self,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        rhs_iter.for_each(|(rhs, res, grad, lhs_sum_grad)| {
            self.eq_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn ne_backward_lhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, lhs_sum_grad)| {
            self.ne_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn ne_backward_rhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, rhs_sum_grad)| {
            self.ne_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad)
        });
    }
    #[inline]
    fn ne_backward_lhs_iter_fix_rhs<'a>(
        &self,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        lhs_iter.for_each(|(lhs, res, grad, lhs_sum_grad)| {
            self.ne_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn ne_backward_rhs_iter_fix_lhs<'a>(
        &self,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        rhs_iter.for_each(|(rhs, res, grad, lhs_sum_grad)| {
            self.ne_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn le_backward_lhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, lhs_sum_grad)| {
            self.le_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn le_backward_rhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, rhs_sum_grad)| {
            self.le_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad)
        });
    }
    #[inline]
    fn le_backward_lhs_iter_fix_rhs<'a>(
        &self,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        lhs_iter.for_each(|(lhs, res, grad, lhs_sum_grad)| {
            self.le_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn le_backward_rhs_iter_fix_lhs<'a>(
        &self,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        rhs_iter.for_each(|(rhs, res, grad, lhs_sum_grad)| {
            self.le_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn ge_backward_lhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, lhs_sum_grad)| {
            self.ge_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn ge_backward_rhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, rhs_sum_grad)| {
            self.ge_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad)
        });
    }
    #[inline]
    fn ge_backward_lhs_iter_fix_rhs<'a>(
        &self,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        lhs_iter.for_each(|(lhs, res, grad, lhs_sum_grad)| {
            self.ge_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn ge_backward_rhs_iter_fix_lhs<'a>(
        &self,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        rhs_iter.for_each(|(rhs, res, grad, lhs_sum_grad)| {
            self.ge_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn lt_backward_lhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, lhs_sum_grad)| {
            self.lt_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn lt_backward_rhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, rhs_sum_grad)| {
            self.lt_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad)
        });
    }
    #[inline]
    fn lt_backward_lhs_iter_fix_rhs<'a>(
        &self,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        lhs_iter.for_each(|(lhs, res, grad, lhs_sum_grad)| {
            self.lt_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn lt_backward_rhs_iter_fix_lhs<'a>(
        &self,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        rhs_iter.for_each(|(rhs, res, grad, lhs_sum_grad)| {
            self.lt_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn gt_backward_lhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, lhs_sum_grad)| {
            self.gt_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn gt_backward_rhs_iter<'a>(
        &self,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        iter.for_each(|(lhs, rhs, res, grad, rhs_sum_grad)| {
            self.gt_backward_rhs(lhs, rhs, res, grad, rhs_sum_grad)
        });
    }
    #[inline]
    fn gt_backward_lhs_iter_fix_rhs<'a>(
        &self,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        lhs_iter.for_each(|(lhs, res, grad, lhs_sum_grad)| {
            self.gt_backward_lhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
    #[inline]
    fn gt_backward_rhs_iter_fix_lhs<'a>(
        &self,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        rhs_iter.for_each(|(rhs, res, grad, lhs_sum_grad)| {
            self.gt_backward_rhs(lhs, rhs, res, grad, lhs_sum_grad)
        });
    }
}

impl DiscreteBinaryIter for GradMethodDiscrete {}
impl DiscreteBinaryIter for GradMethodLinear {}
impl DiscreteBinaryIter for GradMethodSigmoid {}

const CMP_METHOD_DISCRET: GradMethodDiscrete = GradMethodDiscrete;

impl DiscreteBinaryOpT for super::op::Eq {
    const OP: DiscreteBinaryOp = DiscreteBinaryOp::Eq;
    #[inline]
    fn forward(lhs: f64, rhs: f64) -> f64 {
        if OrderedFloat(lhs).eq(&OrderedFloat(rhs)) {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.eq_backward_lhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.eq_backward_lhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.eq_backward_lhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.eq_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.eq_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.eq_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.eq_backward_rhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.eq_backward_rhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.eq_backward_rhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.eq_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.eq_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.eq_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl DiscreteBinaryOpT for super::op::Ne {
    const OP: DiscreteBinaryOp = DiscreteBinaryOp::Ne;
    #[inline]
    fn forward(lhs: f64, rhs: f64) -> f64 {
        if OrderedFloat(lhs).ne(&OrderedFloat(rhs)) {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ne_backward_lhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.ne_backward_lhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ne_backward_lhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ne_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.ne_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ne_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ne_backward_rhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.ne_backward_rhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ne_backward_rhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ne_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.ne_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ne_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl DiscreteBinaryOpT for super::op::Le {
    const OP: DiscreteBinaryOp = DiscreteBinaryOp::Le;
    #[inline]
    fn forward(lhs: f64, rhs: f64) -> f64 {
        if OrderedFloat(lhs).le(&OrderedFloat(rhs)) {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.le_backward_lhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.le_backward_lhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.le_backward_lhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.le_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.le_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.le_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.le_backward_rhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.le_backward_rhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.le_backward_rhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.le_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.le_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.le_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl DiscreteBinaryOpT for super::op::Ge {
    const OP: DiscreteBinaryOp = DiscreteBinaryOp::Ge;
    #[inline]
    fn forward(lhs: f64, rhs: f64) -> f64 {
        if OrderedFloat(lhs).ge(&OrderedFloat(rhs)) {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ge_backward_lhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.ge_backward_lhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ge_backward_lhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ge_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.ge_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ge_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ge_backward_rhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.ge_backward_rhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ge_backward_rhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.ge_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.ge_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.ge_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl DiscreteBinaryOpT for super::op::Lt {
    const OP: DiscreteBinaryOp = DiscreteBinaryOp::Lt;
    #[inline]
    fn forward(lhs: f64, rhs: f64) -> f64 {
        if OrderedFloat(lhs).lt(&OrderedFloat(rhs)) {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.lt_backward_lhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.lt_backward_lhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.lt_backward_lhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.lt_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.lt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.lt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.lt_backward_rhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.lt_backward_rhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.lt_backward_rhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.lt_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.lt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.lt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl DiscreteBinaryOpT for super::op::Gt {
    const OP: DiscreteBinaryOp = DiscreteBinaryOp::Gt;
    #[inline]
    fn forward(lhs: f64, rhs: f64) -> f64 {
        if OrderedFloat(lhs).gt(&OrderedFloat(rhs)) {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.gt_backward_lhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.gt_backward_lhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.gt_backward_lhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        grad_method: &GradMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.gt_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.gt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.gt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        grad_method: &GradMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.gt_backward_rhs_iter(iter),
            GradMethod::Linear(grad_method_linear) => grad_method_linear.gt_backward_rhs_iter(iter),
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.gt_backward_rhs_iter(iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        grad_method: &GradMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match grad_method {
            GradMethod::Discrete => CMP_METHOD_DISCRET.gt_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            GradMethod::Linear(grad_method_linear) => {
                grad_method_linear.gt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            GradMethod::Sigmoid(grad_method_sigmoid) => {
                grad_method_sigmoid.gt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}
