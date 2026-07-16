use crate::Tensor;

#[derive(Clone, Debug, Default)]
pub struct SoftmaxCrossEntropy {
    probabilities: Option<Tensor>,
    labels: Option<Vec<usize>>,
}

impl SoftmaxCrossEntropy {
    pub fn new() -> Self {
        Self {
            probabilities: None,
            labels: None,
        }
    }

    pub fn forward(&mut self, logits: &Tensor, labels: &[usize]) -> f32 {
        assert_eq!(
            logits.rank(),
            2,
            "softmax cross-entropy logits must be a matrix"
        );
        let batch_size = logits.shape()[0];
        let class_count = logits.shape()[1];
        assert_eq!(
            labels.len(),
            batch_size,
            "received {} labels for a batch containing {} examples",
            labels.len(),
            batch_size
        );
        let mut probabilities = Tensor::zeros(logits.shape());
        let mut total_loss = 0.0;
        for row in 0..batch_size {
            let label = labels[row];
            assert!(
                label < class_count,
                "label {} is outside the valid class range 0..{}",
                label,
                class_count
            );
            let row_start = row * class_count;
            let mut maximum = logits.data()[row_start];
            for column in 1..class_count {
                maximum = maximum.max(logits.data()[row_start + column]);
            }
            let mut exponential_sum = 0.0;
            for column in 0..class_count {
                let exponential = (logits.data()[row_start + column] - maximum).exp();
                probabilities.data_mut()[row_start + column] = exponential;
                exponential_sum += exponential;
            }
            for column in 0..class_count {
                probabilities.data_mut()[row_start + column] /= exponential_sum;
            }
            let correct_logit = logits.data()[row_start + label];
            total_loss += maximum + exponential_sum.ln() - correct_logit;
        }
        self.probabilities = Some(probabilities);
        self.labels = Some(labels.to_vec());
        total_loss / batch_size as f32
    }

    pub fn backward(&self) -> Tensor {
        let probabilities = self
            .probabilities
            .as_ref()
            .expect("SoftmaxCrossEntropy::backward requires forward to run first");
        let labels = self
            .labels
            .as_ref()
            .expect("SoftmaxCrossEntropy::backward requires forward to run first");
        let batch_size = probabilities.shape()[0];
        let class_count = probabilities.shape()[1];
        let mut gradients = probabilities.clone();
        for row in 0..batch_size {
            let correct_index = row * class_count + labels[row];
            gradients.data_mut()[correct_index] -= 1.0;
        }
        gradients.multiply_scalar_in_place(1.0 / batch_size as f32);
        gradients
    }

    pub fn probabilities(&self) -> &Tensor {
        self.probabilities
            .as_ref()
            .expect("probabilities are unavailable before forward runs")
    }
}
