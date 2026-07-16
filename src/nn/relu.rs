use crate::Tensor;

#[derive(Clone, Debug, Default)]
pub struct ReLU {
    positive_mask: Vec<u8>,
    cached_shape: Vec<usize>,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            positive_mask: Vec::new(),
            cached_shape: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        if self.positive_mask.len() != input.len() {
            self.positive_mask.resize(input.len(), 0);
        }
        self.cached_shape.clear();
        self.cached_shape.extend_from_slice(input.shape());

        let mut output = Tensor::zeros(input.shape());
        for index in 0..input.len() {
            if input.data()[index] > 0.0 {
                output.data_mut()[index] = input.data()[index];
                self.positive_mask[index] = 1;
            } else {
                self.positive_mask[index] = 0;
            }
        }
        output
    }

    pub fn forward_inference(&self, input: &Tensor) -> Tensor {
        let mut output = Tensor::zeros(input.shape());
        for index in 0..input.len() {
            output.data_mut()[index] = input.data()[index].max(0.0);
        }
        output
    }

    pub fn backward(&self, output_gradients: &Tensor) -> Tensor {
        assert!(
            !self.cached_shape.is_empty(),
            "ReLU::backward requires ReLU::forward to run first"
        );
        assert_eq!(
            output_gradients.shape(),
            self.cached_shape.as_slice(),
            "ReLU gradients have shape {:?}, but cached input has shape {:?}",
            output_gradients.shape(),
            self.cached_shape
        );
        let mut input_gradients = Tensor::zeros(output_gradients.shape());
        for index in 0..output_gradients.len() {
            if self.positive_mask[index] == 1 {
                input_gradients.data_mut()[index] = output_gradients.data()[index];
            }
        }
        input_gradients
    }
}
