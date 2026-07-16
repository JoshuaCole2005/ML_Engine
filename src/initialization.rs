use crate::{Rng, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Initialization {
    XavierUniform,
    XavierNormal,
    HeUniform,
    HeNormal,
}

pub fn initialize_weights(
    input_size: usize,
    output_size: usize,
    initialization: Initialization,
    rng: &mut Rng,
) -> Tensor {
    assert!(input_size > 0, "input size must be greater than zero");
    assert!(output_size > 0, "output size must be greater than zero");
    let mut data = vec![0.0; input_size * output_size];
    match initialization {
        Initialization::XavierUniform => {
            let limit = (6.0 / (input_size + output_size) as f32).sqrt();
            for value in &mut data {
                *value = rng.uniform(-limit, limit);
            }
        }
        Initialization::XavierNormal => {
            let standard_deviation = (2.0 / (input_size + output_size) as f32).sqrt();
            for value in &mut data {
                *value = rng.normal(0.0, standard_deviation);
            }
        }
        Initialization::HeUniform => {
            let limit = (6.0 / input_size as f32).sqrt();
            for value in &mut data {
                *value = rng.uniform(-limit, limit);
            }
        }
        Initialization::HeNormal => {
            let standard_deviation = (2.0 / input_size as f32).sqrt();
            for value in &mut data {
                *value = rng.normal(0.0, standard_deviation);
            }
        }
    }
    Tensor::from_vec(data, &[input_size, output_size])
}
