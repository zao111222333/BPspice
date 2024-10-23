mod expression;

use pyo3::prelude::*;

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
    m.add_function(wrap_pyfunction!(expression::before_update, m)?)?;
    m.add_class::<expression::Expression>()?;
    m.add_class::<Ckt>()?;
    Ok(())
}
