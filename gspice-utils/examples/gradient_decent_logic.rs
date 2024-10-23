use gspice_utils::expression::{before_update, Expression};
fn main() {
    let len = 200;
    let iter = 1000;
    let step = 0.01;
    let (a, a_ref) = Expression::rand_uniform(len, -1.0, 1.0, true);
    let (b, b_ref) = Expression::rand_uniform(len, -1.0, 1.0, true);
    let one = Expression::constant(1.);
    let zero = Expression::constant(0.);
    let f = a.eq_sigmoid(&b, 2.0).cond(&one, &zero);
    let f_loss = a.sub(&b).abs();
    let mut loss = f64::MAX;
    println!("To maximize f = (a==b)? 1 : 0");
    println!("we want a and b as close to each other as possible");
    println!("BEGIN\n  a {a}\n  b {b}");
    for i in 0..iter {
        if i % 40 == 0 {
            f.value();
            let new_loss = f_loss.value().overall_sum() / len as f64;
            assert!(new_loss < loss);
            loss = new_loss;
            println!("iter {i}; loss = avg|a-b| = {loss:5e}");
        }
        let grads = f.backward();
        let df_da = grads.get(&a_ref).unwrap();
        let df_db = grads.get(&b_ref).unwrap();
        before_update();
        a_ref.update_callback(&df_da, |d: &f64| step * d);
        b_ref.update_callback(&df_db, |d: &f64| step * d);
    }
    let loss = f_loss.value().overall_sum() / len as f64;
    println!("iter {iter}; loss = avg|a-b| = {loss:5e}");
    println!("END\n  a {a}\n  b {b}");
}
