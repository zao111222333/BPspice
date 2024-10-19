use core::fmt;
use std::ops::Deref;

use ordered_float::OrderedFloat;

use super::{Expression, ScalarTensor, Tensor};

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.values().read().unwrap(), f)
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Const(v) => write!(f, "Const({})", v),
            Expression::Parameter(tensor) | Expression::Operation(tensor, _) => {
                write!(f, "Tensor({})", tensor)
            }
        }
    }
}

// impl PartialEq for Tensor {
//     fn eq(&self, other: &Self) -> bool {
//         let lhs_v = self.values().read().unwrap();
//         let rhs_v = other.values().read().unwrap();
//         lhs_v.len() == rhs_v.len()
//             && lhs_v
//                 .deref()
//                 .iter()
//                 .zip(rhs_v.deref().iter())
//                 .all(|(x1, x2)| OrderedFloat(*x1).eq(&OrderedFloat(*x2)))
//     }
// }

impl<'a> ScalarTensor<'a> {
    pub fn eq_vec(&self, values: &[f64]) -> bool {
        match self {
            ScalarTensor::Tensor(tensor) => {
                let lhs_v = tensor.values().read().unwrap();
                lhs_v.len() == values.len()
                    && lhs_v
                        .deref()
                        .iter()
                        .zip(values.iter())
                        .all(|(x1, x2)| OrderedFloat(*x1).eq(&OrderedFloat(*x2)))
            }
            _ => false,
        }
    }
    pub fn eq_num(&self, value: f64) -> bool {
        match self {
            ScalarTensor::Scalar(f) => value.eq(*f),
            _ => false,
        }
    }
}
