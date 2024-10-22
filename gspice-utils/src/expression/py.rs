use core::fmt;

use pyo3::{exceptions::PyException, prelude::*, types::PyType};

use super::{autograd::Grad, Expression, ScalarTensor, TensorRef, impls::fmt_vec};

#[pyclass(name = "ScalarTensor")]
#[derive(Clone, Debug)]
enum PyScalarTensor {
    Scalar(f64),
    Tensor(Vec<f64>),
}

#[pymethods]
impl Expression {
    #[pyo3(name = "constant")]
    #[classmethod]
    #[inline]
    fn py_constant(_cls: &Bound<'_, PyType>, value: f64) -> Self {
        Self::constant(value)
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
impl TensorRef {
    #[pyo3(name = "update")]
    fn py_update(&self, grad: &Grad, call_back: Bound<'_, PyAny>) -> PyResult<()> {
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

#[pymethods]
impl PyScalarTensor {
    #[inline]
    fn __repr__(&self) -> String {
        match self {
            Self::Scalar(x) => format!("Const({x})"),
            Self::Tensor(tensor) => {
                struct T<'a>(&'a [f64]);
                impl<'a> fmt::Display for T<'a> {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result{
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
    fn __add__(&self, rhs: &Expression) -> Expression {
        self.add(rhs)
    }
    #[inline]
    fn __sub__(&self, rhs: &Expression) -> Expression {
        self.sub(rhs)
    }
    #[inline]
    fn __mul__(&self, rhs: &Expression) -> Expression {
        self.mul(rhs)
    }
    #[inline]
    fn __div__(&self, rhs: &Expression) -> Expression {
        self.div(rhs)
    }
    // TODO: why it needs 2 args
    #[inline]
    fn __pow__(&self, rhs: &Expression, _mod: bool) -> Expression {
        self.pow(rhs)
    }
    #[inline]
    fn __max__(&self, rhs: &Expression) -> Expression {
        self.max(rhs)
    }
    #[inline]
    fn __min__(&self, rhs: &Expression) -> Expression {
        self.min(rhs)
    }
    #[inline]
    fn __and__(&self, rhs: &Expression) -> Expression {
        self.logic_and(rhs)
    }
    #[inline]
    fn __or__(&self, rhs: &Expression) -> Expression {
        self.logic_or(rhs)
    }
    #[inline]
    fn __not__(&self) -> Expression {
        self.logic_not()
    }
    #[inline]
    fn __neg__(&self) -> Expression {
        self.neg()
    }
    #[inline]
    fn __abs__(&self) -> Expression {
        self.abs()
    }
    #[inline]
    fn __eq__(&self, rhs: &Expression) -> Expression {
        self.eq(rhs)
    }
    #[inline]
    fn __ne__(&self, rhs: &Expression) -> Expression {
        self.ne(rhs)
    }
    #[inline]
    fn __le__(&self, rhs: &Expression) -> Expression {
        self.le(rhs)
    }
    #[inline]
    fn __lt__(&self, rhs: &Expression) -> Expression {
        self.le(rhs)
    }
    #[inline]
    fn __ge__(&self, rhs: &Expression) -> Expression {
        self.ge(rhs)
    }
    #[inline]
    fn __gt__(&self, rhs: &Expression) -> Expression {
        self.gt(rhs)
    }
}
