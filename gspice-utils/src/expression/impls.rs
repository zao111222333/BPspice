use super::{
    op::{CmpOp, CmpOpT, CmpMethod, CmpMethodLinear, CmpMethodDiscret, CmpMethodSigmoid, CmpMethodT},
    Expression, ScalarTensor, Tensor,
};
use core::fmt;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.values().read().unwrap(), f)
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Const(v) => write!(f, "Const({})", v),
            Expression::Tensor(tensor) => {
                write!(f, "Tensor({})", tensor)
            }
        }
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
}
impl<'a> fmt::Display for ScalarTensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarTensor::Scalar(v) => write!(f, "Scalar({})", v),
            ScalarTensor::Tensor(tensor) => {
                write!(f, "Tensor({:?})", tensor.read().unwrap())
            }
        }
    }
}

pub(super) trait CmpIter: CmpMethodT {
    #[inline]
    fn eq_forward_iter<'a>(&self, iter: impl Iterator<Item = (&'a f64, &'a f64)>) -> Vec<f64> {
        iter.map(|(lhs, rhs)| self.eq_forward(*lhs, *rhs)).collect()
    }
    #[inline]
    fn eq_forward_iter_fix_rhs<'a>(
        &self,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        lhs_iter
            .map(move |lhs| self.eq_forward(*lhs, rhs))
            .collect()
    }
    #[inline]
    fn eq_forward_iter_fix_lhs<'a>(
        &self,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        rhs_iter
            .map(move |rhs| self.eq_forward(lhs, *rhs))
            .collect()
    }
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
    fn ne_forward_iter<'a>(&self, iter: impl Iterator<Item = (&'a f64, &'a f64)>) -> Vec<f64> {
        iter.map(|(lhs, rhs)| self.ne_forward(*lhs, *rhs)).collect()
    }
    #[inline]
    fn ne_forward_iter_fix_rhs<'a>(
        &self,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        lhs_iter
            .map(move |lhs| self.ne_forward(*lhs, rhs))
            .collect()
    }
    #[inline]
    fn ne_forward_iter_fix_lhs<'a>(
        &self,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        rhs_iter
            .map(move |rhs| self.ne_forward(lhs, *rhs))
            .collect()
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
    fn le_forward_iter<'a>(&self, iter: impl Iterator<Item = (&'a f64, &'a f64)>) -> Vec<f64> {
        iter.map(|(lhs, rhs)| self.le_forward(*lhs, *rhs)).collect()
    }
    #[inline]
    fn le_forward_iter_fix_rhs<'a>(
        &self,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        lhs_iter
            .map(move |lhs| self.le_forward(*lhs, rhs))
            .collect()
    }
    #[inline]
    fn le_forward_iter_fix_lhs<'a>(
        &self,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        rhs_iter
            .map(move |rhs| self.le_forward(lhs, *rhs))
            .collect()
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
    fn ge_forward_iter<'a>(&self, iter: impl Iterator<Item = (&'a f64, &'a f64)>) -> Vec<f64> {
        iter.map(|(lhs, rhs)| self.ge_forward(*lhs, *rhs)).collect()
    }
    #[inline]
    fn ge_forward_iter_fix_rhs<'a>(
        &self,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        lhs_iter
            .map(move |lhs| self.ge_forward(*lhs, rhs))
            .collect()
    }
    #[inline]
    fn ge_forward_iter_fix_lhs<'a>(
        &self,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        rhs_iter
            .map(move |rhs| self.ge_forward(lhs, *rhs))
            .collect()
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
    fn lt_forward_iter<'a>(&self, iter: impl Iterator<Item = (&'a f64, &'a f64)>) -> Vec<f64> {
        iter.map(|(lhs, rhs)| self.lt_forward(*lhs, *rhs)).collect()
    }
    #[inline]
    fn lt_forward_iter_fix_rhs<'a>(
        &self,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        lhs_iter
            .map(move |lhs| self.lt_forward(*lhs, rhs))
            .collect()
    }
    #[inline]
    fn lt_forward_iter_fix_lhs<'a>(
        &self,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        rhs_iter
            .map(move |rhs| self.lt_forward(lhs, *rhs))
            .collect()
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
    fn gt_forward_iter<'a>(&self, iter: impl Iterator<Item = (&'a f64, &'a f64)>) -> Vec<f64> {
        iter.map(|(lhs, rhs)| self.gt_forward(*lhs, *rhs)).collect()
    }
    #[inline]
    fn gt_forward_iter_fix_rhs<'a>(
        &self,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        lhs_iter
            .map(move |lhs| self.gt_forward(*lhs, rhs))
            .collect()
    }
    #[inline]
    fn gt_forward_iter_fix_lhs<'a>(
        &self,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        rhs_iter
            .map(move |rhs| self.gt_forward(lhs, *rhs))
            .collect()
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

impl CmpIter for CmpMethodDiscret {}
impl CmpIter for CmpMethodLinear {}
impl CmpIter for CmpMethodSigmoid {}

const CMP_METHOD_DISCRET: CmpMethodDiscret = CmpMethodDiscret;

impl CmpOpT for super::op::Eq {
    const OP: CmpOp = CmpOp::Eq;
    #[inline]
    fn forward(cmp_method: &CmpMethod, lhs: f64, rhs: f64) -> f64 {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_forward(lhs, rhs),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.eq_forward(lhs, rhs),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.eq_forward(lhs, rhs),
        }
    }
    #[inline]
    fn forward_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64)>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_forward_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.eq_forward_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.eq_forward_iter(iter),
        }
    }
    #[inline]
    fn forward_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_forward_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.eq_forward_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.eq_forward_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
    #[inline]
    fn forward_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_forward_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.eq_forward_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.eq_forward_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_backward_lhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.eq_backward_lhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.eq_backward_lhs_iter(iter),
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.eq_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.eq_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_backward_rhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.eq_backward_rhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.eq_backward_rhs_iter(iter),
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.eq_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.eq_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.eq_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl CmpOpT for super::op::Ne {
    const OP: CmpOp = CmpOp::Ne;
    #[inline]
    fn forward(cmp_method: &CmpMethod, lhs: f64, rhs: f64) -> f64 {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_forward(lhs, rhs),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ne_forward(lhs, rhs),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ne_forward(lhs, rhs),
        }
    }
    #[inline]
    fn forward_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64)>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_forward_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ne_forward_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ne_forward_iter(iter),
        }
    }
    #[inline]
    fn forward_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_forward_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ne_forward_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ne_forward_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
    #[inline]
    fn forward_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_forward_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ne_forward_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ne_forward_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_backward_lhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ne_backward_lhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ne_backward_lhs_iter(iter),
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ne_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ne_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_backward_rhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ne_backward_rhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ne_backward_rhs_iter(iter),
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ne_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ne_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ne_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl CmpOpT for super::op::Le {
    const OP: CmpOp = CmpOp::Le;
    #[inline]
    fn forward(cmp_method: &CmpMethod, lhs: f64, rhs: f64) -> f64 {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_forward(lhs, rhs),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.le_forward(lhs, rhs),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.le_forward(lhs, rhs),
        }
    }
    #[inline]
    fn forward_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64)>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_forward_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.le_forward_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.le_forward_iter(iter),
        }
    }
    #[inline]
    fn forward_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_forward_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.le_forward_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.le_forward_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
    #[inline]
    fn forward_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_forward_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.le_forward_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.le_forward_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_backward_lhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.le_backward_lhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.le_backward_lhs_iter(iter),
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.le_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.le_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_backward_rhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.le_backward_rhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.le_backward_rhs_iter(iter),
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.le_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.le_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.le_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl CmpOpT for super::op::Ge {
    const OP: CmpOp = CmpOp::Ge;
    #[inline]
    fn forward(cmp_method: &CmpMethod, lhs: f64, rhs: f64) -> f64 {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_forward(lhs, rhs),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ge_forward(lhs, rhs),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ge_forward(lhs, rhs),
        }
    }
    #[inline]
    fn forward_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64)>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_forward_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ge_forward_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ge_forward_iter(iter),
        }
    }
    #[inline]
    fn forward_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_forward_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ge_forward_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ge_forward_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
    #[inline]
    fn forward_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_forward_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ge_forward_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ge_forward_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_backward_lhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ge_backward_lhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ge_backward_lhs_iter(iter),
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ge_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ge_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_backward_rhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.ge_backward_rhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.ge_backward_rhs_iter(iter),
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.ge_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.ge_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.ge_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl CmpOpT for super::op::Lt {
    const OP: CmpOp = CmpOp::Lt;
    #[inline]
    fn forward(cmp_method: &CmpMethod, lhs: f64, rhs: f64) -> f64 {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_forward(lhs, rhs),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.lt_forward(lhs, rhs),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.lt_forward(lhs, rhs),
        }
    }
    #[inline]
    fn forward_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64)>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_forward_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.lt_forward_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.lt_forward_iter(iter),
        }
    }
    #[inline]
    fn forward_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_forward_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.lt_forward_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.lt_forward_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
    #[inline]
    fn forward_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_forward_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.lt_forward_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.lt_forward_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_backward_lhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.lt_backward_lhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.lt_backward_lhs_iter(iter),
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.lt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.lt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_backward_rhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.lt_backward_rhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.lt_backward_rhs_iter(iter),
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.lt_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.lt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.lt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}

impl CmpOpT for super::op::Gt {
    const OP: CmpOp = CmpOp::Gt;
    #[inline]
    fn forward(cmp_method: &CmpMethod, lhs: f64, rhs: f64) -> f64 {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_forward(lhs, rhs),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.gt_forward(lhs, rhs),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.gt_forward(lhs, rhs),
        }
    }
    #[inline]
    fn forward_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64)>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_forward_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.gt_forward_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.gt_forward_iter(iter),
        }
    }
    #[inline]
    fn forward_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: f64,
        rhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_forward_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.gt_forward_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.gt_forward_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
    #[inline]
    fn forward_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: f64,
        lhs_iter: impl Iterator<Item = &'a f64>,
    ) -> Vec<f64> {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_forward_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.gt_forward_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.gt_forward_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_lhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_backward_lhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.gt_backward_lhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.gt_backward_lhs_iter(iter),
        }
    }
    #[inline]
    fn backward_lhs_iter_fix_rhs<'a>(
        cmp_method: &CmpMethod,
        rhs: &f64,
        lhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_backward_lhs_iter_fix_rhs(rhs, lhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.gt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.gt_backward_lhs_iter_fix_rhs(rhs, lhs_iter)
            }
        }
    }
    #[inline]
    fn backward_rhs_iter<'a>(
        cmp_method: &CmpMethod,
        iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_backward_rhs_iter(iter),
            CmpMethod::Linear(cmp_method_linear) => cmp_method_linear.gt_backward_rhs_iter(iter),
            CmpMethod::Sigmoid(cmp_method_sigmoid) => cmp_method_sigmoid.gt_backward_rhs_iter(iter),
        }
    }
    #[inline]
    fn backward_rhs_iter_fix_lhs<'a>(
        cmp_method: &CmpMethod,
        lhs: &f64,
        rhs_iter: impl Iterator<Item = (&'a f64, &'a f64, &'a f64, &'a mut f64)>,
    ) {
        match cmp_method {
            CmpMethod::Discret => CMP_METHOD_DISCRET.gt_backward_rhs_iter_fix_lhs(lhs, rhs_iter),
            CmpMethod::Linear(cmp_method_linear) => {
                cmp_method_linear.gt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
            CmpMethod::Sigmoid(cmp_method_sigmoid) => {
                cmp_method_sigmoid.gt_backward_rhs_iter_fix_lhs(lhs, rhs_iter)
            }
        }
    }
}
