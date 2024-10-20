// use std::{collections::HashMap, rc::Arc, sync::{Arc, RwLock}};
// use core::ops::*;
// enum UnaryOp {
//     Neg,
//     Powf(f64),
// }

// enum BinaryOp {
//     Add,
//     Sub,
//     Mul,
//     Div,
//     Pow,
// }

// struct Value {
//     value: f64,
//     grads: Option<Vec<f64>>,
// }

// impl Add for Value {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         let value = self.value + other.value;

//         let derivative = self
//             .grads
//             .iter()
//             .zip(other.grads.iter())
//             .map(|(a, b)| a + b)
//             .collect();

//         Dual { value, derivative }
//     }
// }

// enum Expression {
//     Value(Value),
//     ValueSwipe(Arc<RwLock<Value>>),
//     UnaryOp(UnaryOp, Box<Expression>),
//     BinaryOp(BinaryOp, Box<Expression>, Box<Expression>),
// }

// impl Expression {
//     fn value(&self) -> Value {
//         match self {
//             Self::Value(ref value) => value,
//             Self::Neg(ref value) => value.0.neg(),
//             Self::Add(rc) => todo!(),
//             Self::Sub(rc) => todo!(),
//             Self::Mul(rc) => todo!(),
//             Self::Div(rc) => todo!(),
//         }
//     }
// }
