use crate::{Dense, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    learning_rate: f32,
}

impl Sgd {
    pub fn new(learning_rate: f32) -> Self {
        assert!(
            learning_rate > 0.0,
            "learning rate must be greater than zero"
        );
        Self { learning_rate }
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn step_dense(&self, layer: &mut Dense) {
        layer.apply_sgd(self.learning_rate);
    }
}

#[derive(Clone, Debug)]
pub struct AdamState {
    weight_first_moment: Tensor,
    weight_second_moment: Tensor,
    bias_first_moment: Tensor,
    bias_second_moment: Tensor,
    step: u64,
}

impl AdamState {
    pub fn for_dense(layer: &Dense) -> Self {
        Self {
            weight_first_moment: Tensor::zeros(layer.weights().shape()),
            weight_second_moment: Tensor::zeros(layer.weights().shape()),
            bias_first_moment: Tensor::zeros(layer.biases().shape()),
            bias_second_moment: Tensor::zeros(layer.biases().shape()),
            step: 0,
        }
    }

    pub fn step(&self) -> u64 {
        self.step
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        assert!(
            learning_rate > 0.0,
            "learning rate must be greater than zero"
        );
        assert!(beta1 >= 0.0 && beta1 < 1.0, "beta1 must be in [0, 1)");
        assert!(beta2 >= 0.0 && beta2 < 1.0, "beta2 must be in [0, 1)");
        assert!(epsilon > 0.0, "epsilon must be greater than zero");
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }

    pub fn standard(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8)
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn step_dense(&self, layer: &mut Dense, state: &mut AdamState) {
        assert_eq!(
            layer.weights().shape(),
            state.weight_first_moment.shape(),
            "Adam weight state does not match dense layer"
        );
        assert_eq!(
            layer.biases().shape(),
            state.bias_first_moment.shape(),
            "Adam bias state does not match dense layer"
        );

        state.step += 1;
        let step = state.step as f32;
        let first_correction = 1.0 - self.beta1.powf(step);
        let second_correction = 1.0 - self.beta2.powf(step);

        for index in 0..layer.weights().len() {
            let gradient = layer.weight_gradients().data()[index];

            let previous_first = state.weight_first_moment.data()[index];
            let previous_second = state.weight_second_moment.data()[index];

            state.weight_first_moment.data_mut()[index] =
                self.beta1 * previous_first + (1.0 - self.beta1) * gradient;

            state.weight_second_moment.data_mut()[index] =
                self.beta2 * previous_second + (1.0 - self.beta2) * gradient * gradient;

            let corrected_first = state.weight_first_moment.data()[index] / first_correction;
            let corrected_second = state.weight_second_moment.data()[index] / second_correction;
            let update =
                self.learning_rate * corrected_first / (corrected_second.sqrt() + self.epsilon);

            layer.weights_mut().data_mut()[index] -= update;
        }

        for index in 0..layer.biases().len() {
            let gradient = layer.bias_gradients().data()[index];

            let previous_first = state.bias_first_moment.data()[index];
            let previous_second = state.bias_second_moment.data()[index];

            state.bias_first_moment.data_mut()[index] =
                self.beta1 * previous_first + (1.0 - self.beta1) * gradient;

            state.bias_second_moment.data_mut()[index] =
                self.beta2 * previous_second + (1.0 - self.beta2) * gradient * gradient;

            let corrected_first = state.bias_first_moment.data()[index] / first_correction;
            let corrected_second = state.bias_second_moment.data()[index] / second_correction;
            let update =
                self.learning_rate * corrected_first / (corrected_second.sqrt() + self.epsilon);

            layer.biases_mut().data_mut()[index] -= update;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Adam, AdamState};
    use crate::{Dense, Initialization, Rng, Tensor};

    #[test]
    fn adam_first_step_uses_bias_corrected_moments() {
        let mut rng = Rng::new(1);
        let mut layer = Dense::new(1, 1, Initialization::XavierUniform, &mut rng);
        *layer.weights_mut() = Tensor::from_vec(vec![2.0], &[1, 1]);
        *layer.biases_mut() = Tensor::from_vec(vec![1.0], &[1]);

        let input = Tensor::from_vec(vec![3.0], &[1, 1]);
        layer.forward(&input);
        layer.backward(&Tensor::from_vec(vec![4.0], &[1, 1]));

        let optimizer = Adam::standard(0.1);
        let mut state = AdamState::for_dense(&layer);
        optimizer.step_dense(&mut layer, &mut state);

        assert!((layer.weights().data()[0] - 1.9).abs() < 0.00001);
        assert!((layer.biases().data()[0] - 0.9).abs() < 0.00001);
        assert_eq!(state.step(), 1);
    }
}
