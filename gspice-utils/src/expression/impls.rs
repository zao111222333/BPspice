use super::{Expression, ScalarTensor, Tensor};
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
