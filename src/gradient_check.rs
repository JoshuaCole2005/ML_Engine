use crate::{Dense, Initialization, Rng, Tensor};

pub fn check_dense_weight_gradient() -> f32 {
    let mut rng = Rng::new(5);
    let mut layer = Dense::new(2, 2, Initialization::XavierUniform, &mut rng);
    *layer.weights_mut() = Tensor::from_vec(vec![0.2, -0.3, 0.4, 0.1], &[2, 2]);
    *layer.biases_mut() = Tensor::from_vec(vec![0.0, 0.0], &[2]);
    let input = Tensor::from_vec(vec![0.5, -1.0, 1.5, 2.0], &[2, 2]);
    let upstream = Tensor::from_vec(vec![0.7, -0.2, -0.4, 0.9], &[2, 2]);
    layer.forward(&input);
    layer.backward(&upstream);
    let analytical = layer.weight_gradients().data()[0];
    let epsilon = 0.001;
    let original = layer.weights().data()[0];
    layer.weights_mut().data_mut()[0] = original + epsilon;
    let plus = dot(&layer.forward_inference(&input), &upstream);
    layer.weights_mut().data_mut()[0] = original - epsilon;
    let minus = dot(&layer.forward_inference(&input), &upstream);
    layer.weights_mut().data_mut()[0] = original;
    let numerical = (plus - minus) / (2.0 * epsilon);
    (analytical - numerical).abs()
}

fn dot(left: &Tensor, right: &Tensor) -> f32 {
    assert_eq!(left.shape(), right.shape(), "dot requires matching shapes");
    let mut result = 0.0;
    for index in 0..left.len() {
        result += left.data()[index] * right.data()[index];
    }
    result
}
