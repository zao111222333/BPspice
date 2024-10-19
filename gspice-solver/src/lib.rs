use candle_core::{Device, IndexOp, Result, Tensor, Var};
pub fn add(left: &Var, right: &Tensor) -> Result<Tensor> {
    // a.i(1)?;
    // let c = a.matmul(&b)?;
    let out = left.i(0)?.add(&right.i(0)?)?;
    out.is_variable();
    let grads = out.sum_all()?.backward()?;
    let d_out_d_left = grads.get(&left).unwrap();
    println!("d_out_d_left {d_out_d_left}");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let device = Device::Cpu;
        let a = Var::new(&[1f32, 2f32, 3f32], &device).unwrap();
        let b = Tensor::new(&[1f32, 2f32, 3f32], &device).unwrap();
        let (storage, layout) = b.storage_and_layout();
        // storage
        let result = add(&a, &b).unwrap();
        println!("{}", result.eq(2f32).unwrap());
    }
}
