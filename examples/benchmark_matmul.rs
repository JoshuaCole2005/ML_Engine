use std::time::Instant;

use ml_engine::{Rng, Tensor};

fn main() {
    let mut rng = Rng::new(1);
    let left_data: Vec<f32> = (0..128 * 784).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let right_data: Vec<f32> = (0..784 * 128).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let left = Tensor::from_vec(left_data, &[128, 784]);
    let right = Tensor::from_vec(right_data, &[784, 128]);
    let mut output = Tensor::zeros(&[128, 128]);

    left.matmul_into(&right, &mut output);

    let repetitions = 10;
    let start = Instant::now();
    for _ in 0..repetitions {
        left.matmul_into(&right, &mut output);
    }
    let elapsed = start.elapsed();

    println!(
        "{} blocked matrix multiplications: {:.2?}",
        repetitions, elapsed
    );
    println!("Checksum: {:.5}", output.sum());
}
