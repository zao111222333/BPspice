use core::fmt;

use pyo3::{exceptions::PyException, prelude::*, types::PyType};

use super::{autograd::Grad, impls::fmt_vec, Expression, ScalarTensor, TensorRef};

#[pyclass]
struct Tensor(gspice::Tensor);

#[pyclass]
pub struct TensorRef(gspice::TensorRef);

#[pymethods]
impl TensorRef {
    /// Need [`before_update`] before calling this
    ///
    /// Need [`Expression::value`](Expression::value) after calling this
    ///
    /// Tensor = values
    #[inline]
    pub fn assign(&self, values: Vec<f64>) {
        self.0.assign(values);
    }
    fn update(&self, grad: &Grad, call_back: Bound<'_, PyAny>) -> PyResult<()> {
        if call_back.is_callable() {
            let f = move |x: &f64| -> PyResult<f64> {
                // Acquire the GIL
                Python::with_gil(|py| {
                    // Convert the Rust f64 to a Python object
                    let arg = x.to_object(py);
                    // Call the Python function with the argument
                    let result = call_back.call1((arg,))?;
                    // Try to extract the result as f64
                    let output: f64 = result.extract()?;
                    Ok(output)
                })
            };
            f(&0.0)?;
            self.update_callback(&grad, |x| f(x).unwrap());
            Ok(())
        } else {
            Err(PyException::new_err("Provided object is not callable"))
        }
    }
}

#[pyclass]
struct Grad(gspice::Grad);
#[pymethods]
impl Grad {
    fn value(&self) -> Vec<f64> {
        self.0.clone()
    }
    fn __repr__(&self) -> String {
        self.0.to_string()
    }
}

#[pyclass]
#[derive(Debug)]
struct GradStore(gspice::GradStore);

#[pymethods]
impl GradStore {
    /// Remove & take the gradient tensor associated with the given tensor-reference
    pub fn take(&mut self, tensor_ref: &TensorRef) -> Option<Grad> {
        if let Some(grad_id) = tensor_ref.0.grad_id() {
            self.0.remove(grad_id)
        } else {
            panic!("The tensor is not with gradient")
        }
    }
}

#[pymethods]
impl Expression {
    /// When you update the compute graph's tensor value.
    /// You need [self.value](Expression::value) before
    /// run [self.backward](Expression::backward) to update its compute graph's value
    fn backward(&self) -> GradStore {
        GradStore(self.backward())
    }
}

#[pyclass]
struct Expression(gspice::Expression);

#[pymethods]
impl Expression {
    #[pyo3(name = "constant")]
    #[classmethod]
    #[inline]
    fn constant(_cls: &Bound<'_, PyType>, value: f64) -> Self {
        Self(gspice::Expression::constant(value))
    }
    #[pyo3(name = "tensor")]
    #[classmethod]
    #[inline]
    fn py_tensor(_cls: &Bound<'_, PyType>, values: Vec<f64>, need_grad: bool) -> (Self, TensorRef) {
        Self::tensor(values, need_grad)
    }
    #[pyo3(name = "zeros")]
    #[classmethod]
    #[inline]
    fn py_zeros(_cls: &Bound<'_, PyType>, len: usize, need_grad: bool) -> (Self, TensorRef) {
        Self::zeros(len, need_grad)
    }
    #[pyo3(name = "ones")]
    #[classmethod]
    #[inline]
    fn py_ones(_cls: &Bound<'_, PyType>, len: usize, need_grad: bool) -> (Self, TensorRef) {
        Self::ones(len, need_grad)
    }
    #[pyo3(name = "rand_uniform")]
    #[classmethod]
    #[inline]
    fn py_rand_uniform(
        _cls: &Bound<'_, PyType>,
        len: usize,
        lower: f64,
        upper: f64,
        need_grad: bool,
    ) -> (Self, TensorRef) {
        Self::rand_uniform(len, lower, upper, need_grad)
    }
    #[pyo3(name = "rand_bernoulli")]
    #[classmethod]
    #[inline]
    fn py_rand_bernoulli(
        _cls: &Bound<'_, PyType>,
        len: usize,
        p: f64,
        need_grad: bool,
    ) -> (Self, TensorRef) {
        Self::rand_bernoulli(len, p, need_grad)
    }
    #[pyo3(name = "value")]
    #[inline]
    fn py_value<'a>(&'a self) -> PyScalarTensor {
        match self.recompute().into() {
            ScalarTensor::Scalar(x) => PyScalarTensor::Scalar(*x),
            ScalarTensor::Tensor(tensor) => PyScalarTensor::Tensor(tensor.read().unwrap().clone()),
        }
    }
    #[inline]
    fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[pymethods]
impl PyScalarTensor {
    #[inline]
    fn __repr__(&self) -> String {
        match self {
            Self::Scalar(x) => format!("Const({x})"),
            Self::Tensor(tensor) => {
                struct T<'a>(&'a [f64]);
                impl<'a> fmt::Display for T<'a> {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(f, "Tensor")?;
                        fmt_vec(self.0, f)
                    }
                }
                T(&tensor).to_string()
            }
        }
    }
}

#[pymethods]
impl Expression {
    #[inline]
    fn __add__(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }
    #[inline]
    fn __sub__(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }
    #[inline]
    fn __mul__(&self, rhs: &Self) -> Self {
        self.mul(rhs)
    }
    #[inline]
    fn __div__(&self, rhs: &Self) -> Self {
        self.div(rhs)
    }
    // TODO: why it needs 2 args
    #[inline]
    fn __pow__(&self, rhs: &Self, _mod: bool) -> Self {
        self.pow(rhs)
    }
    #[inline]
    fn __max__(&self, rhs: &Self) -> Self {
        self.max(rhs)
    }
    #[inline]
    fn __min__(&self, rhs: &Self) -> Self {
        self.min(rhs)
    }
    #[inline]
    fn __and__(&self, rhs: &Self) -> Self {
        self.logic_and(rhs)
    }
    #[inline]
    fn __or__(&self, rhs: &Self) -> Self {
        self.logic_or(rhs)
    }
    #[inline]
    fn __not__(&self) -> Self {
        self.logic_not()
    }
    #[inline]
    fn __neg__(&self) -> Self {
        self.neg()
    }
    #[inline]
    fn __abs__(&self) -> Self {
        self.abs()
    }
    #[inline]
    fn __eq__(&self, rhs: &Self) -> Self {
        self.eq(rhs)
    }
    #[inline]
    fn __ne__(&self, rhs: &Self) -> Self {
        self.ne(rhs)
    }
    #[inline]
    fn __le__(&self, rhs: &Self) -> Self {
        self.le(rhs)
    }
    #[inline]
    fn __lt__(&self, rhs: &Self) -> Self {
        self.le(rhs)
    }
    #[inline]
    fn __ge__(&self, rhs: &Self) -> Self {
        self.ge(rhs)
    }
    #[inline]
    fn __gt__(&self, rhs: &Self) -> Self {
        self.gt(rhs)
    }
    #[inline]
    pub fn cond(&self, on_true: &Self, on_false: &Self) -> Self {}
}

#[pymethods]
impl Expression {
    #[inline]
    pub fn neg(&self) -> Self {
        Self::unary_op::<Neg>(&self)
    }
    #[inline]
    pub fn sin(&self) -> Self {
        Self::unary_op::<Sin>(&self)
    }
    #[inline]
    pub fn cos(&self) -> Self {
        Self::unary_op::<Cos>(&self)
    }
    #[inline]
    pub fn tanh(&self) -> Self {
        Self::unary_op::<Tanh>(&self)
    }
    #[inline]
    pub fn tan(&self) -> Self {
        Self::unary_op::<Tan>(&self)
    }
    #[inline]
    pub fn ceil(&self) -> Self {
        Self::unary_op::<Ceil>(&self)
    }
    #[inline]
    pub fn floor(&self) -> Self {
        Self::unary_op::<Floor>(&self)
    }
    #[inline]
    pub fn round(&self) -> Self {
        Self::unary_op::<Round>(&self)
    }
    #[inline]
    pub fn sign(&self) -> Self {
        Self::unary_op::<Sign>(&self)
    }
    #[inline]
    pub fn sqrt(&self) -> Self {
        Self::unary_op::<Sqrt>(&self)
    }
    #[inline]
    pub fn sqr(&self) -> Self {
        Self::unary_op::<Sqr>(&self)
    }
    #[inline]
    pub fn cubic(&self) -> Self {
        Self::unary_op::<Cubic>(&self)
    }
    #[inline]
    pub fn log(&self) -> Self {
        Self::unary_op::<Log>(&self)
    }
    #[inline]
    pub fn exp(&self) -> Self {
        Self::unary_op::<Exp>(&self)
    }
    #[inline]
    pub fn abs(&self) -> Self {
        Self::unary_op::<Abs>(&self)
    }
    #[inline]
    pub fn erf(&self) -> Self {
        Self::unary_op::<Erf>(&self)
    }
    #[inline]
    pub fn logic_not(&self) -> Self {
        Self::unary_op::<LogicNot>(&self)
    }
}

#[pymethods]
impl Expression {
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        self.binary_op::<Add>(rhs)
    }
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        self.binary_op::<Sub>(rhs)
    }
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        self.binary_op::<Mul>(rhs)
    }
    #[inline]
    pub fn div(&self, rhs: &Self) -> Self {
        self.binary_op::<Div>(rhs)
    }
    #[inline]
    pub fn pow(&self, rhs: &Self) -> Self {
        self.binary_op::<Pow>(rhs)
    }
    #[inline]
    pub fn min(&self, rhs: &Self) -> Self {
        self.binary_op::<Min>(rhs)
    }
    #[inline]
    pub fn max(&self, rhs: &Self) -> Self {
        self.binary_op::<Max>(rhs)
    }
    #[inline]
    pub fn logic_and(&self, rhs: &Self) -> Self {
        self.binary_op::<LogicAnd>(rhs)
    }
    #[inline]
    pub fn logic_or(&self, rhs: &Self) -> Self {
        self.binary_op::<LogicOr>(rhs)
    }
}

#[pymethods]
impl Expression {
    #[inline]
    pub fn eq(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Eq>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn ne(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Ne>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn le(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Le>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn ge(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Ge>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn lt(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Lt>(rhs, GradMethod::Discrete)
    }
    #[inline]
    pub fn gt(&self, rhs: &Self) -> Self {
        self.discrete_binary_op::<Gt>(rhs, GradMethod::Discrete)
    }
    /// `eq(a,b) = sigmoid(a, b, k) = e^(-k (a - b)^2)`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn eq_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Eq>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `ne(a,b) = 1- sigmoid(a, b, k) = 1-e^(-k (a - b)^2)`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn ne_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Ne>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `le(a,b) = 1 / (1 + e^(k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn le_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Le>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `ge(a,b) = 1 / (1 + e^(-k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn ge_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Ge>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `lt(a,b) = 1 / (1 + e^(k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn lt_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Lt>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `gt(a,b) = 1 / (1 + e^(-k(a - b)))`
    ///
    /// **only activate when graident is required!**
    #[inline]
    pub fn gt_sigmoid(&self, rhs: &Self, k: f64) -> Self {
        self.discrete_binary_op::<Gt>(rhs, GradMethod::new_sigmoid(k))
    }
    /// `1 - |a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    ///                1
    ///       /\       
    ///      /  \
    /// ____/    \___  0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn eq_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Eq>(rhs, GradMethod::new_linear(epsilon))
    }
    /// |`a - b|/ε`    when  `|a - b| < ε`
    /// ``` text
    /// ___      ____    1
    ///    \    /        
    ///     \  /
    ///      \/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn ne_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Ne>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 - (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    /// ____           1
    ///     \          
    ///       \
    ///         \___   0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn le_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Le>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 + (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    ///          ____  1
    ///         /      
    ///       /
    /// ____/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn ge_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Ge>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 - (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    /// ____           1
    ///     \          
    ///       \
    ///         \___   0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn lt_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Lt>(rhs, GradMethod::new_linear(epsilon))
    }
    /// `1/2 + (a-b)/2ε`    when  `|a - b| < ε`
    /// ``` text
    ///          ____  1
    ///         /      
    ///       /
    /// ____/          0
    /// --------------->
    ///   -ε  0  ε     a-b
    /// ```
    /// **only activate when graident is required!**
    #[inline]
    pub fn gt_linear(&self, rhs: &Self, epsilon: f64) -> Self {
        self.discrete_binary_op::<Gt>(rhs, GradMethod::new_linear(epsilon))
    }
}

#[pyfunction]
pub fn before_update() {}

#[pyclass(name = "ScalarTensor")]
#[derive(Clone, Debug)]
enum PyScalarTensor {
    Scalar(f64),
    Tensor(Vec<f64>),
}
