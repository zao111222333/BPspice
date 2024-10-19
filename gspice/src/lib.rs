mod expression;

pub use expression::{ChangeMarker, Expression, ScalarTensor, Tensor};

use std::sync::{Arc, RwLock};

// use candle_core::{Tensor, Storage, Device, Var};

// pub struct Value(pub Tensor);
// pub struct Parameter(pub Arc<RwLock<f64>>);

// pub const DEVICE: Device = Device::Cpu;
// impl Value {
//     pub fn new(f: f64) -> Result<Self, candle_core::Error> {
//         Ok(Self(Tensor::new(f, &DEVICE)?))
//     }
//     pub fn new_grad(f: f64) -> Result<Self, candle_core::Error> {
//         Ok(Self(Var::new(f, &DEVICE)?.into_inner()))
//     }
//     pub fn parameter(f: f64) -> Result<(Self)>
// }

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
