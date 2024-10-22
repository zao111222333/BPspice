enum Optimizer {
    Linear { step: f64 },
    Adam { step: f64 },
}

impl Optimizer {
    fn next_epoch(&mut self) {}
    fn gradient_decent(&self) -> impl Fn(f64) -> f64 {
        |x| x
    }
}
