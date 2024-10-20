use std::{
    collections::{BTreeMap, HashMap},
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use super::{op::BinaryOp, Expression, Op, Tensor, TensorRef};
use core::cmp::Ordering;

#[derive(Debug)]
pub struct Grad(Vec<f64>);
impl Grad {
    pub fn inner(self) -> Vec<f64> {
        self.0
    }
}

impl Deref for Grad {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Grad {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct GradId(usize);

impl PartialOrd for GradId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GradId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0).reverse()
    }
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);
impl GradId {
    // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
    pub(super) fn new() -> Self {
        Self(COUNTER.fetch_add(1, Relaxed))
    }
}

/// A store for gradients, associating a scalar id to the corresponding gradient scalar, used for back propagation.
#[derive(Debug)]
pub struct GradStore(HashMap<GradId, Grad>);

impl Expression {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    fn sorted_nodes(&self) -> BTreeMap<GradId, &Tensor> {
        fn walk<'a>(expr: &'a Expression, already_seen: &mut BTreeMap<GradId, &'a Tensor>) {
            if let Expression::Tensor(tensor) = expr {
                if let Some(grad_id) = tensor.grad_id() {
                    if already_seen.get(grad_id).is_none() {
                        already_seen.insert(*grad_id, &tensor);
                        match tensor.op() {
                            Op::Assgin => (),
                            Op::Powf(expr, _) => walk(expr, already_seen),
                            Op::Cond(cond, when_true, when_false) => todo!(),
                            Op::Cmp(expr, cmp_op) => todo!(),
                            Op::Unary(expr, unary_op) => walk(expr, already_seen),
                            Op::Binary(expr1, expr2, _) => {
                                walk(expr1, already_seen);
                                walk(expr2, already_seen);
                            }
                        }
                    }
                }
            }
        }
        let mut already_seen = BTreeMap::new();
        walk(self, &mut already_seen);
        already_seen
    }
    pub fn backward(&self) -> GradStore {
        let sorted_nodes = self.sorted_nodes();
        if let Some((first_id, first_tensor)) = sorted_nodes.first_key_value() {
            let mut grads = GradStore::new();
            grads.insert(*first_id, Grad(first_tensor.ones_like()));
            for (grad_id, tensor) in sorted_nodes {
                if let Op::Assgin = tensor.op() {
                    continue;
                }
                let grad = grads
                    .remove_id(&grad_id)
                    .expect("gspice internal error - grad not populated");
                match tensor.op() {
                    Op::Assgin => unreachable!(),
                    Op::Powf(expression, _) => todo!(),
                    Op::Cond(cond, when_true, when_false) => todo!(),
                    Op::Cmp(expression, cmp_op) => todo!(),
                    Op::Unary(node, unary_op) => todo!(),
                    Op::Binary(lhs, rhs, binary_op) => {
                        binary_op.backward(lhs, rhs, &mut grads, grad);
                    }
                }
            }
            grads
        } else {
            GradStore::new()
        }
    }
}

impl GradStore {
    /// Create a new gradient store
    fn new() -> Self {
        GradStore(HashMap::new())
    }

    /// Get the gradient tensor associated with the given tensor-reference
    pub fn get(&self, tensor_ref: &TensorRef) -> Option<&Grad> {
        if let Some(grad_id) = tensor_ref.0.grad_id() {
            self.0.get(grad_id)
        } else {
            panic!("The tensor is not with gradient")
        }
    }

    /// Remove & take the gradient tensor associated with the given tensor-reference
    pub fn remove(&mut self, tensor_ref: &TensorRef) -> Option<Grad> {
        if let Some(grad_id) = tensor_ref.0.grad_id() {
            self.0.remove(grad_id)
        } else {
            panic!("The tensor is not with gradient")
        }
    }

    /// Remove the gradient tensor associated with the given tensor, returning it if it exists
    fn remove_id(&mut self, id: &GradId) -> Option<Grad> {
        self.0.remove(id)
    }

    /// Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
    fn insert(&mut self, id: GradId, grad: Grad) -> Option<Grad> {
        self.0.insert(id, grad)
    }

    /// Get the gradient tensor associated with the given tensor, or, if it does not exist,
    /// insert a tensor of zeroes, with the same shape and type as the given tensors and return it
    fn or_insert(&mut self, tensor: &Tensor) -> Option<&mut Grad> {
        use std::collections::hash_map::Entry;
        tensor.grad_id().map(|id| match self.0.entry(id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(Grad(tensor.zeros_like())),
        })
    }
}

impl BinaryOp {
    fn backward(&self, lhs: &Expression, rhs: &Expression, grads: &mut GradStore, grad: Grad) {
        let [fn_backward_lhs, fn_backward_rhs] = self.fn_backward();
        match (lhs, rhs) {
            (Expression::Const(_), Expression::Const(_)) => unreachable!(),
            (Expression::Const(lhs_x), Expression::Tensor(rhs_tensor)) => {
                if let Some(rhs_sum_grad) = grads.or_insert(rhs_tensor) {
                    for (rhs_grad, grad_x, rhs_x) in itertools::izip!(
                        rhs_sum_grad.iter_mut(),
                        grad.iter(),
                        rhs_tensor.values().read().unwrap().iter(),
                    ) {
                        fn_backward_rhs(*lhs_x, *rhs_x, *grad_x, rhs_grad);
                    }
                }
            }
            (Expression::Tensor(lhs_tensor), Expression::Const(rhs_x)) => {
                if let Some(lhs_sum_grad) = grads.or_insert(lhs_tensor) {
                    for (lhs_grad, grad_x, lhs_x) in itertools::izip!(
                        lhs_sum_grad.iter_mut(),
                        grad.iter(),
                        lhs_tensor.values().read().unwrap().iter(),
                    ) {
                        fn_backward_lhs(*lhs_x, *rhs_x, *grad_x, lhs_grad);
                    }
                }
            }
            (Expression::Tensor(lhs_tensor), Expression::Tensor(rhs_tensor)) => {
                if let Some(rhs_sum_grad) = grads.or_insert(rhs_tensor) {
                    for (rhs_grad, grad_x, lhs_x, rhs_x) in itertools::izip!(
                        rhs_sum_grad.iter_mut(),
                        grad.iter(),
                        lhs_tensor.values().read().unwrap().iter(),
                        rhs_tensor.values().read().unwrap().iter(),
                    ) {
                        fn_backward_rhs(*lhs_x, *rhs_x, *grad_x, rhs_grad);
                    }
                }
                if let Some(lhs_sum_grad) = grads.or_insert(lhs_tensor) {
                    for (lhs_grad, grad_x, lhs_x, rhs_x) in itertools::izip!(
                        lhs_sum_grad.iter_mut(),
                        grad.iter(),
                        lhs_tensor.values().read().unwrap().iter(),
                        rhs_tensor.values().read().unwrap().iter(),
                    ) {
                        fn_backward_lhs(*lhs_x, *rhs_x, *grad_x, lhs_grad);
                    }
                }
            }
        }
    }
}
