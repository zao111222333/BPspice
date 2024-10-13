use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn add(a: usize, b: usize) -> PyResult<usize> {
    Ok(bpspice::add(a, b))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "bpspice")]
fn pymodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
