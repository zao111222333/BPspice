use itertools::izip;
use std::{
    collections::{BTreeMap, HashMap},
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use super::{
    op::{BinaryOp, Cond, DiscreteBinaryOp, GradMethod, Powf, UnaryOp},
    Expression, Op, Tensor, TensorRef,
};
use core::cmp::Ordering;

#[derive(Debug)]
pub struct Grad(pub(super) Vec<f64>);
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
    fn grad_walk<'a>(&'a self, already_seen: &mut BTreeMap<GradId, &'a Tensor>) {
        if let Expression::Tensor(tensor) = self {
            if let Some(grad_id) = tensor.grad_id() {
                if already_seen.get(grad_id).is_none() {
                    already_seen.insert(*grad_id, &tensor);
                    match tensor.op() {
                        Op::Assgin => (),
                        Op::Powf(node, _) => node.grad_walk(already_seen),
                        Op::Cond(cond, on_true, on_false) => {
                            cond.grad_walk(already_seen);
                            on_true.grad_walk(already_seen);
                            on_false.grad_walk(already_seen);
                        }
                        Op::Unary(node, _) => node.grad_walk(already_seen),
                        Op::Binary(lhs, rhs, _) | Op::DiscreteBinary(lhs, rhs, _, _) => {
                            lhs.grad_walk(already_seen);
                            rhs.grad_walk(already_seen);
                        }
                    }
                }
            }
        }
    }
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    fn sorted_nodes(&self) -> BTreeMap<GradId, &Tensor> {
        let mut already_seen = BTreeMap::new();
        self.grad_walk(&mut already_seen);
        already_seen
    }
}

impl Expression {
    /// When you update the compute graph's tensor value.
    /// You need [self.value](Expression::value) before
    /// run [self.backward](Expression::backward) to update its compute graph's value
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
                    Op::Powf(node, n) => Powf::_backward(*n, tensor, node, &mut grads, grad),
                    Op::Cond(cond, on_true, on_false) => {
                        Cond::_backward(cond, on_true, on_false, &mut grads, grad)
                    }
                    Op::Unary(node, unary_op) => {
                        unary_op._backward(tensor, node, &mut grads, grad);
                    }
                    Op::Binary(lhs, rhs, binary_op) => {
                        binary_op._backward(tensor, lhs, rhs, &mut grads, grad);
                    }
                    Op::DiscreteBinary(lhs, rhs, discrete_binary_op, grad_method) => {
                        discrete_binary_op._backward(
                            tensor,
                            lhs,
                            rhs,
                            grad_method,
                            &mut grads,
                            grad,
                        )
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

impl UnaryOp {
    fn _backward(&self, tensor: &Tensor, node: &Expression, grads: &mut GradStore, grad: Grad) {
        let backward = self.backward();
        match node {
            Expression::Const(_) => unreachable!(),
            Expression::Tensor(node_tensor) => {
                if let Some(node_sum_grad) = grads.or_insert(node_tensor) {
                    for (sum_grad, res, x, grad) in itertools::izip!(
                        node_sum_grad.iter_mut(),
                        tensor.values().read().unwrap().iter(),
                        node_tensor.values().read().unwrap().iter(),
                        grad.iter(),
                    ) {
                        backward(x, res, grad, sum_grad);
                    }
                }
            }
        }
    }
}

impl Powf {
    fn _backward(n: f64, tensor: &Tensor, node: &Expression, grads: &mut GradStore, grad: Grad) {
        match node {
            Expression::Const(_) => unreachable!(),
            Expression::Tensor(node_tensor) => {
                if let Some(node_sum_grad) = grads.or_insert(node_tensor) {
                    for (sum_grad, res, x, grad) in itertools::izip!(
                        node_sum_grad.iter_mut(),
                        tensor.values().read().unwrap().iter(),
                        node_tensor.values().read().unwrap().iter(),
                        grad.iter(),
                    ) {
                        Self::backward(x, n, res, grad, sum_grad);
                    }
                }
            }
        }
    }
}

impl Cond {
    #[rustfmt::skip]
    fn _backward(
        cond: &Expression,
        on_true: &Expression,
        on_false: &Expression,
        grads: &mut GradStore,
        grad: Grad,
    ) {
        match (cond, on_true, on_false){
            (Expression::Const(_), Expression::Const(_), Expression::Const(_)) => unreachable!(),
            (Expression::Const(cond_x), Expression::Const(on_true_x), Expression::Tensor(on_false_tensor)) => {
                if let Some(on_false_sum_grad) = grads.or_insert(on_false_tensor) {
                    for (on_false_grad, grad, on_false_x) in itertools::izip!(
                        on_false_sum_grad.iter_mut(),
                        grad.iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_false(cond_x, on_true_x, on_false_x, grad, on_false_grad);
                    }
                }
            },
            (Expression::Const(cond_x), Expression::Tensor(on_true_tensor), Expression::Const(on_false_x)) => {
                if let Some(on_true_sum_grad) = grads.or_insert(on_true_tensor) {
                    for (on_true_grad, grad, on_true_x) in itertools::izip!(
                        on_true_sum_grad.iter_mut(),
                        grad.iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_true(cond_x, on_true_x, on_false_x, grad, on_true_grad);
                    }
                }
            },
            (Expression::Const(cond_x), Expression::Tensor(on_true_tensor), Expression::Tensor(on_false_tensor)) => {
                if let Some(on_true_sum_grad) = grads.or_insert(on_true_tensor) {
                    for (on_true_grad, grad, on_true_x, on_false_x) in itertools::izip!(
                        on_true_sum_grad.iter_mut(),
                        grad.iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_true(cond_x, on_true_x, on_false_x, grad, on_true_grad);
                    }
                }
                if let Some(on_false_sum_grad) = grads.or_insert(on_false_tensor) {
                    for (on_false_grad, grad, on_true_x, on_false_x) in itertools::izip!(
                        on_false_sum_grad.iter_mut(),
                        grad.iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_false(cond_x, on_true_x, on_false_x, grad, on_false_grad);
                    }
                }
            },
            (Expression::Tensor(cond_tensor), Expression::Const(on_true_x), Expression::Const(on_false_x)) => {
                if let Some(cond_sum_grad) = grads.or_insert(cond_tensor) {
                    for (cond_grad, grad, cond_x) in itertools::izip!(
                        cond_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_cond(cond_x, on_true_x, on_false_x, grad, cond_grad);
                    }
                }
            },
            (Expression::Tensor(cond_tensor), Expression::Const(on_true_x), Expression::Tensor(on_false_tensor)) => {
                if let Some(cond_sum_grad) = grads.or_insert(cond_tensor) {
                    for (cond_grad, grad, cond_x, on_false_x) in itertools::izip!(
                        cond_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_cond(cond_x, on_true_x, on_false_x, grad, cond_grad);
                    }
                }
                if let Some(on_false_sum_grad) = grads.or_insert(on_false_tensor) {
                    for (on_false_grad, grad, cond_x, on_false_x) in itertools::izip!(
                        on_false_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_false(cond_x, on_true_x, on_false_x, grad, on_false_grad);
                    }
                }
            },
            (Expression::Tensor(cond_tensor), Expression::Tensor(on_true_tensor), Expression::Const(on_false_x)) => {
                if let Some(cond_sum_grad) = grads.or_insert(cond_tensor) {
                    for (cond_grad, grad, cond_x, on_true_x) in itertools::izip!(
                        cond_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_cond(cond_x, on_true_x, on_false_x, grad, cond_grad);
                    }
                }
                if let Some(on_true_sum_grad) = grads.or_insert(on_true_tensor) {
                    for (on_true_grad, grad, cond_x, on_true_x) in itertools::izip!(
                        on_true_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_true(cond_x, on_true_x, on_false_x, grad, on_true_grad);
                    }
                }
            },
            (Expression::Tensor(cond_tensor), Expression::Tensor(on_true_tensor), Expression::Tensor(on_false_tensor)) => {
                if let Some(cond_sum_grad) = grads.or_insert(cond_tensor) {
                    for (cond_grad, grad, cond_x, on_true_x, on_false_x) in itertools::izip!(
                        cond_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_cond(cond_x, on_true_x, on_false_x, grad, cond_grad);
                    }
                }
                if let Some(on_true_sum_grad) = grads.or_insert(on_true_tensor) {
                    for (on_true_grad, grad, cond_x, on_true_x, on_false_x) in itertools::izip!(
                        on_true_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_true(cond_x, on_true_x, on_false_x, grad, on_true_grad);
                    }
                }
                if let Some(on_false_sum_grad) = grads.or_insert(on_false_tensor) {
                    for (on_false_grad, grad, cond_x, on_true_x, on_false_x) in itertools::izip!(
                        on_false_sum_grad.iter_mut(),
                        grad.iter(),
                        cond_tensor.values().read().unwrap().iter(),
                        on_true_tensor.values().read().unwrap().iter(),
                        on_false_tensor.values().read().unwrap().iter(),
                    ) {
                        Self::backward_on_false(cond_x, on_true_x, on_false_x, grad, on_false_grad);
                    }
                }
            },
        }
    }
}

impl DiscreteBinaryOp {
    fn _backward(
        &self,
        tensor: &Tensor,
        lhs: &Expression,
        rhs: &Expression,
        grad_method: &GradMethod,
        grads: &mut GradStore,
        grad: Grad,
    ) {
        // lhs: &f64,
        // rhs: &f64,
        // res: &f64,
        // grad: &f64,
        // rhs_sum_grad: &mut f64,
        match (lhs, rhs) {
            (Expression::Const(_), Expression::Const(_)) => unreachable!(),
            (Expression::Const(lhs_x), Expression::Tensor(rhs_tensor)) => {
                if let Some(rhs_sum_grad) = grads.or_insert(rhs_tensor) {
                    self.backward_rhs_iter_fix_lhs(
                        grad_method,
                        lhs_x,
                        izip!(
                            rhs_tensor.values().read().unwrap().iter(),
                            tensor.values().read().unwrap().iter(),
                            grad.iter(),
                            rhs_sum_grad.iter_mut(),
                        ),
                    );
                }
            }
            (Expression::Tensor(lhs_tensor), Expression::Const(rhs_x)) => {
                if let Some(lhs_sum_grad) = grads.or_insert(lhs_tensor) {
                    self.backward_lhs_iter_fix_rhs(
                        grad_method,
                        rhs_x,
                        izip!(
                            lhs_tensor.values().read().unwrap().iter(),
                            tensor.values().read().unwrap().iter(),
                            grad.iter(),
                            lhs_sum_grad.iter_mut(),
                        ),
                    );
                }
            }
            (Expression::Tensor(lhs_tensor), Expression::Tensor(rhs_tensor)) => {
                if let Some(rhs_sum_grad) = grads.or_insert(rhs_tensor) {
                    self.backward_rhs_iter(
                        grad_method,
                        izip!(
                            lhs_tensor.values().read().unwrap().iter(),
                            rhs_tensor.values().read().unwrap().iter(),
                            tensor.values().read().unwrap().iter(),
                            grad.iter(),
                            rhs_sum_grad.iter_mut(),
                        ),
                    );
                }
                if let Some(lhs_sum_grad) = grads.or_insert(lhs_tensor) {
                    self.backward_lhs_iter(
                        grad_method,
                        izip!(
                            lhs_tensor.values().read().unwrap().iter(),
                            rhs_tensor.values().read().unwrap().iter(),
                            tensor.values().read().unwrap().iter(),
                            grad.iter(),
                            lhs_sum_grad.iter_mut(),
                        ),
                    );
                }
            }
        }
    }
}

impl BinaryOp {
    fn _backward(
        &self,
        tensor: &Tensor,
        lhs: &Expression,
        rhs: &Expression,
        grads: &mut GradStore,
        grad: Grad,
    ) {
        let [backward_lhs, backward_rhs] = self.backward();
        match (lhs, rhs) {
            (Expression::Const(_), Expression::Const(_)) => unreachable!(),
            (Expression::Const(lhs_x), Expression::Tensor(rhs_tensor)) => {
                if let Some(rhs_sum_grad) = grads.or_insert(rhs_tensor) {
                    for (rhs_grad, res, grad, rhs_x) in itertools::izip!(
                        rhs_sum_grad.iter_mut(),
                        tensor.values().read().unwrap().iter(),
                        grad.iter(),
                        rhs_tensor.values().read().unwrap().iter(),
                    ) {
                        backward_rhs(lhs_x, rhs_x, res, grad, rhs_grad);
                    }
                }
            }
            (Expression::Tensor(lhs_tensor), Expression::Const(rhs_x)) => {
                if let Some(lhs_sum_grad) = grads.or_insert(lhs_tensor) {
                    for (lhs_grad, res, grad, lhs_x) in itertools::izip!(
                        lhs_sum_grad.iter_mut(),
                        tensor.values().read().unwrap().iter(),
                        grad.iter(),
                        lhs_tensor.values().read().unwrap().iter(),
                    ) {
                        backward_lhs(lhs_x, rhs_x, res, grad, lhs_grad);
                    }
                }
            }
            (Expression::Tensor(lhs_tensor), Expression::Tensor(rhs_tensor)) => {
                if let Some(rhs_sum_grad) = grads.or_insert(rhs_tensor) {
                    for (rhs_grad, res, grad, lhs_x, rhs_x) in itertools::izip!(
                        rhs_sum_grad.iter_mut(),
                        tensor.values().read().unwrap().iter(),
                        grad.iter(),
                        lhs_tensor.values().read().unwrap().iter(),
                        rhs_tensor.values().read().unwrap().iter(),
                    ) {
                        backward_rhs(lhs_x, rhs_x, res, grad, rhs_grad);
                    }
                }
                if let Some(lhs_sum_grad) = grads.or_insert(lhs_tensor) {
                    for (lhs_grad, res, grad, lhs_x, rhs_x) in itertools::izip!(
                        lhs_sum_grad.iter_mut(),
                        tensor.values().read().unwrap().iter(),
                        grad.iter(),
                        lhs_tensor.values().read().unwrap().iter(),
                        rhs_tensor.values().read().unwrap().iter(),
                    ) {
                        backward_lhs(lhs_x, rhs_x, res, grad, lhs_grad);
                    }
                }
            }
        }
    }
}
