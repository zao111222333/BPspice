use pyo3::prelude::*;
// use std::ops::*;
#[pyclass]
struct Expr(gspice::Expression);

#[pymethods]
impl Expr {
    #[new]
    fn constant(value: f64) -> Self {
        Self(gspice::Expression::constant(value))
    }
    // #[new]
    // fn parameter(values: Vec<f64>, need_grad: bool) -> Self {
    //     let (expr, tensor) = gspice::Expression::parameter(values, need_grad);
    //     Self(expr)
    // }
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }
    fn __add__(&self, other: &Self) -> Self {
        use std::ops::Add;
        Self(self.0.add(&other.0))
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn add(a: usize, b: usize) -> PyResult<usize> {
    Ok(gspice::add(a, b))
}

#[pyclass]
struct Ckt {}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "gspice")]
fn pymodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_class::<Ckt>()?;
    m.add_class::<Expr>()?;
    Ok(())
}
