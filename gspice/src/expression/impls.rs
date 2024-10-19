use core::fmt;
use std::ops::Deref;

use ordered_float::OrderedFloat;

use super::{Expression, Tensor};

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

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let lhs_v = self.values().read().unwrap();
        let rhs_v = other.values().read().unwrap();
        lhs_v.deref().eq(rhs_v.deref())
    }
}

#[cfg(test)]
impl Expression {
    pub(super) fn eq_vec(&self, values: &[f64]) -> bool {
        match self {
            Expression::Parameter(tensor) | Expression::Operation(tensor, _) => {
                tensor.values().read().unwrap().eq(values)
            }
            _ => false,
        }
    }
    pub(super) fn eq_num(&self, value: f64) -> bool {
        match self {
            Expression::Const(f) => value.eq(f),
            _ => false,
        }
    }
}
