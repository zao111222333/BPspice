use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct GradId(usize);

static COUNTER: AtomicUsize = AtomicUsize::new(0);
impl GradId {
    // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
    pub(super) fn new() -> Self {
        Self(COUNTER.fetch_add(1, Relaxed))
    }
}

// impl Expression {
//     fn sorted_nodes(&self) -> Vec<&Expression> {
//         // The vec of sorted nodes is passed as an owned value rather than a mutable reference
//         // to get around some lifetime limitations.
//         fn walk<'a>(
//             node: &'a Expression,
//             nodes: Vec<&'a Expression>,
//             already_seen: &mut HashMap<GradId, bool>,
//         ) -> (bool, Vec<&'a Expression>) {
//             if let Some(&tg) = already_seen.get(&node.id()) {
//                 return (tg, nodes);
//             }
//             let mut track_grad = false;
//             let mut nodes = match &node.0.op {
//                 Op::Constant => nodes,
//                 Op::Variable => {
//                     // Do not call recursively on the "leaf" nodes.
//                     track_grad = true;
//                     nodes
//                 }
//                 Op::Binary(lhs, rhs, _) => {
//                     let (tg, nodes) = walk(lhs, nodes, already_seen);
//                     track_grad |= tg;
//                     let (tg, nodes) = walk(rhs, nodes, already_seen);
//                     track_grad |= tg;
//                     nodes
//                 }
//                 Op::Unary(_node, UnaryOp::Ceil)
//                 | Op::Unary(_node, UnaryOp::Floor)
//                 | Op::Unary(_node, UnaryOp::Round)
//                 | Op::Unary(_node, UnaryOp::Sign) => nodes,
//                 Op::Copy(node) | Op::Cmp(node, _) | Op::Unary(node, _) | Op::Powf(node, _) => {
//                     let (tg, nodes) = walk(node, nodes, already_seen);
//                     track_grad |= tg;
//                     nodes
//                 }
//             };
//             already_seen.insert(node.id(), track_grad);
//             if track_grad {
//                 nodes.push(node);
//             }
//             (track_grad, nodes)
//         }
//         let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
//         nodes.reverse();
//         nodes
//     }
//     pub fn backward(&self) -> GradStore {
//         let sorted_nodes = self.sorted_nodes();
//         let mut grads = GradStore::new();
//         grads.insert(&self, 1.0);
//         for node in sorted_nodes.iter() {
//             if node.is_variable() {
//                 continue;
//             }
//             let grad = grads
//                 .remove(node)
//                 .expect("candle internal error - grad not populated");
//             // // https://github.com/huggingface/candle/issues/1241
//             // // Ideally, we would make these operations in place where possible to ensure that we
//             // // do not have to allocate too often. Here we just call `.detach` to avoid computing
//             // // the backprop graph of the backprop itself. This would be an issue for second order
//             // // derivatives but these are out of scope at the moment.
//             // let do_not_detach = CANDLE_GRAD_DO_NOT_DETACH.with(|b| *b);
//             // let grad = if do_not_detach { grad } else { grad.detach() };
//             // if let Some(op) = node.op() {
//             match &node.0.op {
//                 Op::Binary(lhs, rhs, BinaryOp::Add) => {
//                     let lhs_sum_grad = grads.or_insert(lhs);
//                     *lhs_sum_grad = lhs_sum_grad.add(&grad);
//                     let rhs_sum_grad = grads.or_insert(rhs);
//                     *rhs_sum_grad = rhs_sum_grad.add(&grad);
//                 }
//                 Op::Binary(lhs, rhs, BinaryOp::Sub) => {
//                     let lhs_sum_grad = grads.or_insert(lhs);
//                     *lhs_sum_grad = lhs_sum_grad.add(&grad);
//                     let rhs_sum_grad = grads.or_insert(rhs);
//                     *rhs_sum_grad = rhs_sum_grad.sub(&grad);
//                 }
//                 Op::Binary(lhs, rhs, BinaryOp::Mul) => {
//                     let lhs_grad = grad.mul(rhs.value());
//                     let lhs_sum_grad = grads.or_insert(lhs);
//                     *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad);
//                     let rhs_grad = grad.mul(lhs.value());
//                     let rhs_sum_grad = grads.or_insert(rhs);
//                     *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad);
//                 }
//                 Op::Binary(lhs, rhs, BinaryOp::Div) => {
//                     let lhs_grad = grad.div(rhs.value());
//                     let lhs_sum_grad = grads.or_insert(lhs);
//                     *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad);
//                     let rhs_grad = grad.mul(lhs.value()).div(&rhs.value().sqr());
//                     let rhs_sum_grad = grads.or_insert(rhs);
//                     *rhs_sum_grad = rhs_sum_grad.sub(&rhs_grad);
//                 }
//                 Op::Binary(lhs, rhs, BinaryOp::Minimum)
//                 | Op::Binary(lhs, rhs, BinaryOp::Maximum) => {
//                     let mask_lhs = node.eq(lhs)?.to_dtype(grad.dtype());
//                     let mask_rhs = node.eq(rhs)?.to_dtype(grad.dtype());

//                     // If both masks are 1 one the same point, we want to scale the
//                     // gradient by 0.5 rather than 1.
//                     let lhs_grad = mask_lhs.mul(&grad)?.div(&(&mask_rhs + 1.)?);
//                     let lhs_sum_grad = grads.or_insert(lhs);
//                     *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad);

//                     let rhs_grad = mask_rhs.mul(&grad)?.div(&(&mask_lhs + 1.)?);
//                     let rhs_sum_grad = grads.or_insert(rhs);
//                     *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad);
//                 }
//                 Op::Copy(arg) => {
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&grad)
//                 }
//                 Op::Unary(arg, UnaryOp::Log) => {
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&(grad / arg.value()))
//                 }
//                 Op::Unary(arg, UnaryOp::Sin) => {
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&(&grad * arg.value().cos()))
//                 }
//                 Op::Unary(arg, UnaryOp::Cos) => {
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.sub(&(&grad * arg.value().sin()))
//                 }
//                 Op::Unary(arg, UnaryOp::Tanh) => {
//                     let sum_grad = grads.or_insert(arg);
//                     let minus_dtanh = (node.sqr() - 1.);
//                     *sum_grad = sum_grad.sub(&(&grad * &minus_dtanh))
//                 }
//                 Op::Unary(arg, UnaryOp::Abs) => {
//                     let sum_grad = grads.or_insert(arg);
//                     let ones = 1.0;
//                     let abs_grad = arg.ge(&arg.zeros_like()?)?.where_cond(&ones, &ones.neg()?);
//                     *sum_grad = sum_grad.add(&(&grad * abs_grad)?)?
//                 }
//                 Op::Unary(arg, UnaryOp::Exp) => {
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&(&grad * *node)?)?
//                 }
//                 Op::Unary(arg, UnaryOp::Neg) => {
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.sub(&grad)?
//                 }
//                 Op::Unary(arg, UnaryOp::Recip) => {
//                     let sum_grad = grads.or_insert(arg);
//                     let grad = (grad / arg.sqr()?);
//                     *sum_grad = sum_grad.sub(&grad)?
//                 }
//                 &Op::Narrow(ref arg, dim, start_idx, len) => {
//                     let arg_dims = arg.dims();
//                     let left_pad = if start_idx == 0 {
//                         None
//                     } else {
//                         let mut dims = arg_dims.to_vec();
//                         dims[dim] = start_idx;
//                         Some(Expression::zeros(dims, grad.dtype(), grad.device())?)
//                     };
//                     let right_pad = arg_dims[dim] - start_idx - len;
//                     let right_pad = if right_pad == 0 {
//                         None
//                     } else {
//                         let mut dims = arg_dims.to_vec();
//                         dims[dim] = right_pad;
//                         Some(Expression::zeros(dims, grad.dtype(), grad.device())?)
//                     };
//                     let arg_grad = match (left_pad, right_pad) {
//                         (None, None) => grad,
//                         (Some(l), None) => Expression::cat(&[&l, &grad], dim)?,
//                         (None, Some(r)) => Expression::cat(&[&grad, &r], dim)?,
//                         (Some(l), Some(r)) => Expression::cat(&[&l, &grad, &r], dim)?,
//                     };
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&arg_grad)?
//                 }
//                 Op::Unary(_, UnaryOp::Floor)
//                 | Op::Unary(_, UnaryOp::Round)
//                 | Op::Reduce(_, ReduceOp::ArgMin, _)
//                 | Op::Reduce(_, ReduceOp::ArgMax, _)
//                 | Op::Unary(_, UnaryOp::Sign)
//                 | Op::Cmp(_, _) => {}
//                 Op::Reshape(arg) => {
//                     let arg_grad = grad.reshape(arg.dims());
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&arg_grad)?
//                 }
//                 Op::Unary(_, UnaryOp::Ceil) => Err(Error::BackwardNotSupported { op: "ceil" })?,
//                 Op::Unary(arg, UnaryOp::Gelu) => {
//                     let sum_grad = grads.or_insert(arg);
//                     let cube = arg.powf(3.);
//                     let tanh = (0.0356774 * &cube + (0.797885 * arg)?)?.tanh();
//                     let gelu_grad = (((0.5 * &tanh)?
//                         + (0.0535161 * cube + (0.398942 * arg)?)? * (1. - tanh.powf(2.)?))?
//                         + 0.5);
//                     *sum_grad = sum_grad.add(&(&grad * gelu_grad)?)?
//                 }
//                 Op::Unary(arg, UnaryOp::Erf) => {
//                     let sum_grad = grads.or_insert(arg);
//                     // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
//                     let erf_grad = (2. / std::f64::consts::PI.sqrt()) * (arg.sqr()?.neg()?).exp();
//                     *sum_grad = sum_grad.add(&(&grad * erf_grad)?)?
//                 }
//                 Op::Powf(arg, e) => {
//                     let arg_grad = (&(grad * arg.powf(e - 1.)?)? * *e);
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&arg_grad)?
//                 }
//                 Op::Unary(arg, UnaryOp::Sqr) => {
//                     let arg_grad = arg.mul(&grad)?.affine(2., 0.);
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&arg_grad)?
//                 }
//                 Op::Unary(arg, UnaryOp::Sqrt) => {
//                     let arg_grad = grad.div(node)?.affine(0.5, 0.);
//                     let sum_grad = grads.or_insert(arg);
//                     *sum_grad = sum_grad.add(&arg_grad)?
//                 } // };
//             }
//         }
//         grads
//     }
// }

// /// A store for gradients, associating a scalar id to the corresponding gradient scalar, used for back propagation.
// #[derive(Debug)]
// pub struct GradStore(HashMap<GradId, f64>);

// impl GradStore {
//     /// Create a new gradient store
//     fn new() -> Self {
//         GradStore(HashMap::new())
//     }

//     /// Get the gradient scalar corresponding to the given scalar id
//     pub fn get_id(&self, id: GradId) -> Option<&f64> {
//         self.0.get(&id)
//     }

//     /// Get the gradient scalar associated with the given scalar
//     pub fn get(&self, scalar: &Expression) -> Option<&f64> {
//         self.0.get(&scalar.id())
//     }

//     /// Remove the gradient scalar associated with the given scalar, returning it if it exists
//     pub fn remove(&mut self, scalar: &Expression) -> Option<f64> {
//         self.0.remove(&scalar.id())
//     }

//     /// Insert a gradient scalar associated with the given scalar, returning the previous gradient scalar if it existed
//     pub fn insert(&mut self, scalar: &Expression, grad: f64) -> Option<f64> {
//         self.0.insert(scalar.id(), grad)
//     }

//     /// Get the gradient scalar associated with the given scalar, or, if it does not exist,
//     /// insert a scalar of zeroes, with the same shape and type as the given scalars and return it
//     fn or_insert(&mut self, scalar: &Expression) -> &mut f64 {
//         use std::collections::hash_map::Entry;
//         let grad = match self.0.entry(scalar.id()) {
//             Entry::Occupied(entry) => entry.into_mut(),
//             Entry::Vacant(entry) => entry.insert(0.0),
//         };
//         grad
//     }

//     /// Get the scalar ids of the stored gradient scalars
//     pub fn get_ids(&self) -> impl Iterator<Item = &GradId> {
//         self.0.keys()
//     }
// }

// #[test]
// fn ttt() {
//     // 定义变量 x 和 y
//     let x = Expression::variable(1.0);
//     let y = Expression::variable(2.0);

//     // 定义子表达式 z = x * y
//     let z = ctx.mul(x.clone(), y.clone());

//     // 定义函数 f = z + z
//     let f = ctx.add(z.clone(), z.clone());

//     println!("f = {}", f.value);

//     // 反向传播
//     let mut grad_map = HashMap::new();
//     backward(&f, &mut grad_map, 1.0);

//     // 输出自变量的梯度
//     let grad_x = grad_map.get(&x.id).unwrap_or(&0.0);
//     let grad_y = grad_map.get(&y.id).unwrap_or(&0.0);

//     println!("∂f/∂x = {}", grad_x);
//     println!("∂f/∂y = {}", grad_y);
// }
