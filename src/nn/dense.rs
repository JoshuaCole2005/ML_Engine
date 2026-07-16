use crate::{Initialization, Rng, Tensor, initialize_weights};

#[derive(Clone, Debug)]
pub struct Dense {
    weights: Tensor,
    biases: Tensor,
    weight_gradients: Tensor,
    bias_gradients: Tensor,
    input_cache: Option<Tensor>,
}

impl Dense {
    pub fn new(
        input_size: usize,
        output_size: usize,
        initialization: Initialization,
        rng: &mut Rng,
    ) -> Self {
        let weights = initialize_weights(input_size, output_size, initialization, rng);
        let biases = Tensor::zeros(&[output_size]);
        Self::from_parameters(weights, biases)
    }

    pub fn from_parameters(weights: Tensor, biases: Tensor) -> Self {
        assert_eq!(weights.rank(), 2, "dense weights must be a matrix");
        assert_eq!(biases.rank(), 1, "dense biases must be a vector");
        assert_eq!(
            weights.shape()[1],
            biases.shape()[0],
            "dense weight output size and bias length do not match"
        );
        let weight_gradients = Tensor::zeros(weights.shape());
        let bias_gradients = Tensor::zeros(biases.shape());
        Self {
            weights,
            biases,
            weight_gradients,
            bias_gradients,
            input_cache: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.validate_input(input);
        match &mut self.input_cache {
            Some(cache) if cache.shape() == input.shape() => cache.copy_from(input),
            _ => self.input_cache = Some(input.clone()),
        }
        self.forward_inference(input)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Tensor {
        self.validate_input(input);
        let mut output = input.matmul(&self.weights);
        output.add_row_vector_in_place(&self.biases);
        output
    }

    pub fn backward(&mut self, output_gradients: &Tensor) -> Tensor {
        let input = self
            .input_cache
            .as_ref()
            .expect("Dense::backward requires Dense::forward to run first");
        assert_eq!(
            output_gradients.shape(),
            &[input.shape()[0], self.output_size()],
            "dense output gradients have shape {:?}, but expected [{}, {}]",
            output_gradients.shape(),
            input.shape()[0],
            self.output_size()
        );

        let input_transposed = input.transpose_2d();
        input_transposed.matmul_into(output_gradients, &mut self.weight_gradients);
        output_gradients.sum_rows_into(&mut self.bias_gradients);
        let weights_transposed = self.weights.transpose_2d();
        output_gradients.matmul(&weights_transposed)
    }

    pub fn zero_grad(&mut self) {
        self.weight_gradients.fill(0.0);
        self.bias_gradients.fill(0.0);
    }

    pub(crate) fn apply_sgd(&mut self, learning_rate: f32) {
        for index in 0..self.weights.len() {
            self.weights.data_mut()[index] -= learning_rate * self.weight_gradients.data()[index];
        }
        for index in 0..self.biases.len() {
            self.biases.data_mut()[index] -= learning_rate * self.bias_gradients.data()[index];
        }
    }

    pub fn input_size(&self) -> usize {
        self.weights.shape()[0]
    }

    pub fn output_size(&self) -> usize {
        self.weights.shape()[1]
    }

    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Tensor {
        &mut self.weights
    }

    pub fn biases(&self) -> &Tensor {
        &self.biases
    }

    pub fn biases_mut(&mut self) -> &mut Tensor {
        &mut self.biases
    }

    pub fn weight_gradients(&self) -> &Tensor {
        &self.weight_gradients
    }

    pub fn bias_gradients(&self) -> &Tensor {
        &self.bias_gradients
    }

    fn validate_input(&self, input: &Tensor) {
        assert_eq!(
            input.rank(),
            2,
            "dense input must be a matrix, but received shape {:?}",
            input.shape()
        );
        assert_eq!(
            input.shape()[1],
            self.input_size(),
            "dense input has {} features, but layer expects {}",
            input.shape()[1],
            self.input_size()
        );
    }
}
