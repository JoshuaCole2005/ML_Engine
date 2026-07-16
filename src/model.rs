use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::{Adam, AdamState, Dense, Initialization, ReLU, Rng, Sgd, SoftmaxCrossEntropy, Tensor};

const MODEL_MAGIC: &[u8; 8] = b"MLENG001";

#[derive(Clone, Debug)]
pub struct TwoLayerAdamState {
    dense1: AdamState,
    dense2: AdamState,
}

#[derive(Clone, Debug)]
pub struct TwoLayerNetwork {
    dense1: Dense,
    relu: ReLU,
    dense2: Dense,
    loss_function: SoftmaxCrossEntropy,
}

impl TwoLayerNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, rng: &mut Rng) -> Self {
        Self {
            dense1: Dense::new(input_size, hidden_size, Initialization::HeUniform, rng),
            relu: ReLU::new(),
            dense2: Dense::new(hidden_size, output_size, Initialization::XavierUniform, rng),
            loss_function: SoftmaxCrossEntropy::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let hidden_linear = self.dense1.forward(input);
        let hidden_activated = self.relu.forward(&hidden_linear);
        self.dense2.forward(&hidden_activated)
    }

    pub fn logits(&self, input: &Tensor) -> Tensor {
        let hidden_linear = self.dense1.forward_inference(input);
        let hidden_activated = self.relu.forward_inference(&hidden_linear);
        self.dense2.forward_inference(&hidden_activated)
    }

    pub fn train_batch(&mut self, input: &Tensor, labels: &[usize], optimizer: &Sgd) -> f32 {
        let loss = self.forward_and_backward(input, labels);
        optimizer.step_dense(&mut self.dense1);
        optimizer.step_dense(&mut self.dense2);
        loss
    }

    pub fn train_batch_adam(
        &mut self,
        input: &Tensor,
        labels: &[usize],
        optimizer: &Adam,
        state: &mut TwoLayerAdamState,
    ) -> f32 {
        let loss = self.forward_and_backward(input, labels);
        optimizer.step_dense(&mut self.dense1, &mut state.dense1);
        optimizer.step_dense(&mut self.dense2, &mut state.dense2);
        loss
    }

    fn forward_and_backward(&mut self, input: &Tensor, labels: &[usize]) -> f32 {
        let logits = self.forward(input);
        let loss = self.loss_function.forward(&logits, labels);
        let logits_gradients = self.loss_function.backward();
        let hidden_gradients = self.dense2.backward(&logits_gradients);
        let hidden_linear_gradients = self.relu.backward(&hidden_gradients);
        self.dense1.backward(&hidden_linear_gradients);
        loss
    }

    pub fn loss(&mut self, input: &Tensor, labels: &[usize]) -> f32 {
        let logits = self.logits(input);
        self.loss_function.forward(&logits, labels)
    }

    pub fn evaluate_batch(&mut self, input: &Tensor, labels: &[usize]) -> (f32, usize) {
        let logits = self.logits(input);
        let loss = self.loss_function.forward(&logits, labels);
        let predictions = logits.argmax_rows();
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(prediction, label)| prediction == label)
            .count();
        (loss, correct)
    }

    pub fn predict(&self, input: &Tensor) -> Vec<usize> {
        self.logits(input).argmax_rows()
    }

    pub fn accuracy(&self, input: &Tensor, labels: &[usize]) -> f32 {
        let predictions = self.predict(input);
        assert_eq!(
            predictions.len(),
            labels.len(),
            "prediction and label counts do not match"
        );
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(prediction, label)| prediction == label)
            .count();
        correct as f32 / labels.len() as f32
    }

    pub fn adam_state(&self) -> TwoLayerAdamState {
        TwoLayerAdamState {
            dense1: AdamState::for_dense(&self.dense1),
            dense2: AdamState::for_dense(&self.dense2),
        }
    }

    pub fn zero_grad(&mut self) {
        self.dense1.zero_grad();
        self.dense2.zero_grad();
    }

    pub fn input_size(&self) -> usize {
        self.dense1.input_size()
    }

    pub fn hidden_size(&self) -> usize {
        self.dense1.output_size()
    }

    pub fn output_size(&self) -> usize {
        self.dense2.output_size()
    }

    pub fn dense1(&self) -> &Dense {
        &self.dense1
    }

    pub fn dense1_mut(&mut self) -> &mut Dense {
        &mut self.dense1
    }

    pub fn dense2(&self) -> &Dense {
        &self.dense2
    }

    pub fn dense2_mut(&mut self) -> &mut Dense {
        &mut self.dense2
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MODEL_MAGIC)?;
        write_u64(&mut writer, self.input_size() as u64)?;
        write_u64(&mut writer, self.hidden_size() as u64)?;
        write_u64(&mut writer, self.output_size() as u64)?;
        write_tensor(&mut writer, self.dense1.weights())?;
        write_tensor(&mut writer, self.dense1.biases())?;
        write_tensor(&mut writer, self.dense2.weights())?;
        write_tensor(&mut writer, self.dense2.biases())?;
        writer.flush()
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MODEL_MAGIC {
            return Err(invalid_data("file is not an ML Engine model"));
        }

        let input_size = read_usize(&mut reader, "input size")?;
        let hidden_size = read_usize(&mut reader, "hidden size")?;
        let output_size = read_usize(&mut reader, "output size")?;

        if input_size == 0 || hidden_size == 0 || output_size == 0 {
            return Err(invalid_data("model dimensions must be greater than zero"));
        }

        let dense1_weights = read_tensor(&mut reader, &[input_size, hidden_size])?;
        let dense1_biases = read_tensor(&mut reader, &[hidden_size])?;
        let dense2_weights = read_tensor(&mut reader, &[hidden_size, output_size])?;
        let dense2_biases = read_tensor(&mut reader, &[output_size])?;

        Ok(Self {
            dense1: Dense::from_parameters(dense1_weights, dense1_biases),
            relu: ReLU::new(),
            dense2: Dense::from_parameters(dense2_weights, dense2_biases),
            loss_function: SoftmaxCrossEntropy::new(),
        })
    }
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_usize<R: Read>(reader: &mut R, name: &str) -> io::Result<usize> {
    let value = read_u64(reader)?;
    usize::try_from(value).map_err(|_| invalid_data(&format!("{name} is too large")))
}

fn write_tensor<W: Write>(writer: &mut W, tensor: &Tensor) -> io::Result<()> {
    write_u64(writer, tensor.len() as u64)?;
    for value in tensor.data() {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_tensor<R: Read>(reader: &mut R, shape: &[usize]) -> io::Result<Tensor> {
    let stored_length = read_usize(reader, "tensor length")?;
    let expected_length = shape
        .iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension));
    let expected_length =
        expected_length.ok_or_else(|| invalid_data("tensor shape is too large"))?;

    if stored_length != expected_length {
        return Err(invalid_data(&format!(
            "stored tensor has {stored_length} values, but shape {shape:?} requires {expected_length}"
        )));
    }

    let mut data = Vec::with_capacity(stored_length);
    for _ in 0..stored_length {
        let mut bytes = [0_u8; 4];
        reader.read_exact(&mut bytes)?;
        data.push(f32::from_le_bytes(bytes));
    }
    Ok(Tensor::from_vec(data, shape))
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::TwoLayerNetwork;
    use crate::{Rng, Tensor};

    #[test]
    fn model_save_and_load_preserve_predictions() {
        let mut rng = Rng::new(42);
        let network = TwoLayerNetwork::new(2, 4, 3, &mut rng);
        let input = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let expected = network.logits(&input);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ml_engine_model_{}_{}.bin",
            std::process::id(),
            unique
        ));

        network.save(&path).unwrap();
        let loaded = TwoLayerNetwork::load(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(loaded.input_size(), 2);
        assert_eq!(loaded.hidden_size(), 4);
        assert_eq!(loaded.output_size(), 3);
        assert_eq!(loaded.logits(&input), expected);
    }
}
