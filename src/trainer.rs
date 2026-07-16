use crate::{Adam, MnistDataset, Rng, Tensor, TwoLayerAdamState, TwoLayerNetwork};

#[derive(Clone, Copy, Debug)]
pub struct Evaluation {
    pub loss: f32,
    pub accuracy: f32,
}

pub fn train_epoch_adam(
    network: &mut TwoLayerNetwork,
    dataset: &MnistDataset,
    batch_size: usize,
    optimizer: &Adam,
    optimizer_state: &mut TwoLayerAdamState,
    rng: &mut Rng,
) -> f32 {
    assert!(batch_size > 0, "batch size must be greater than zero");
    assert!(!dataset.is_empty(), "training dataset cannot be empty");

    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    rng.shuffle(&mut indices);

    let mut reusable_inputs = Tensor::zeros(&[batch_size, dataset.image_size()]);
    let mut reusable_labels = Vec::with_capacity(batch_size);
    let mut total_loss = 0.0;
    let mut trained_examples = 0;

    let mut full_chunks = indices.chunks_exact(batch_size);
    for batch_indices in &mut full_chunks {
        dataset.fill_batch(batch_indices, &mut reusable_inputs, &mut reusable_labels);
        let loss = network.train_batch_adam(
            &reusable_inputs,
            &reusable_labels,
            optimizer,
            optimizer_state,
        );
        total_loss += loss * batch_indices.len() as f32;
        trained_examples += batch_indices.len();
    }

    let remainder = full_chunks.remainder();
    if !remainder.is_empty() {
        let (inputs, labels) = dataset.batch(remainder);
        let loss = network.train_batch_adam(&inputs, &labels, optimizer, optimizer_state);
        total_loss += loss * remainder.len() as f32;
        trained_examples += remainder.len();
    }

    total_loss / trained_examples as f32
}

pub fn evaluate(
    network: &mut TwoLayerNetwork,
    dataset: &MnistDataset,
    batch_size: usize,
) -> Evaluation {
    assert!(batch_size > 0, "batch size must be greater than zero");
    assert!(!dataset.is_empty(), "evaluation dataset cannot be empty");

    let mut total_loss = 0.0;
    let mut total_correct = 0;

    for start in (0..dataset.len()).step_by(batch_size) {
        let end = (start + batch_size).min(dataset.len());
        let (inputs, labels) = dataset.batch_range(start, end);
        let (loss, correct) = network.evaluate_batch(&inputs, &labels);
        total_loss += loss * labels.len() as f32;
        total_correct += correct;
    }

    Evaluation {
        loss: total_loss / dataset.len() as f32,
        accuracy: total_correct as f32 / dataset.len() as f32,
    }
}
