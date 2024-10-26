use gspice_utils::expression::{before_update, Expression};

fn main() {
    let len = 2;
    let iter = 1000;
    let step = 0.01;
    let (x, x_ref) = Expression::rand_uniform(len, -1., 1., true);
    let (y, y_ref) = Expression::rand_uniform(len, -1., 1., true);
    let f = &x.sqr() + &y.sqr();
    let mut loss = f64::MAX;
    println!("To minimize f = x^2+y^2");
    println!("we want x and y as close to 0 as possible");
    println!("BEGIN\n  x {x}\n  y {y}");
    for i in 0..iter {
        if i % 40 == 0 {
            let loss_value = f.value();
            let new_loss = loss_value.overall_sum();
            assert!(new_loss < loss);
            loss = new_loss;
            println!("iter {i}; loss = x^2+y^2 = {loss:5e}");
        }
        let grads = f.backward();
        let df_dx = grads.get(&x_ref).unwrap();
        let df_dy = grads.get(&y_ref).unwrap();
        before_update();
        x_ref.update_iter(df_dx.iter().map(|d| -step * d));
        y_ref.update_iter(df_dy.iter().map(|d| -step * d));
    }
    let loss = f.value().overall_sum();
    println!("iter {iter}; loss = x^2+y^2 = {loss:5e}");
    println!("END\n  x {x}\n  y {y}");
}
