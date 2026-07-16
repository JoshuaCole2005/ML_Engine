use std::env;
use std::error::Error;
use std::path::PathBuf;

use ml_engine::{MnistDataset, TwoLayerNetwork, evaluate};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    let data_directory = arguments
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/mnist"));
    let model_path = arguments
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("mnist_model.bin"));

    let test_dataset = MnistDataset::load(
        data_directory.join("t10k-images-idx3-ubyte"),
        data_directory.join("t10k-labels-idx1-ubyte"),
    )?;
    let mut network = TwoLayerNetwork::load(model_path)?;
    let metrics = evaluate(&mut network, &test_dataset, 100);

    println!("Test loss: {:.5}", metrics.loss);
    println!("Test accuracy: {:.2}%", metrics.accuracy * 100.0);
    Ok(())
}
