use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use ml_engine::{Adam, MnistDataset, Rng, TwoLayerNetwork, evaluate, train_epoch_adam};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    let data_directory = arguments
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/mnist"));
    let epochs = parse_usize_argument(&arguments, 2, 5, "epochs")?;
    let batch_size = parse_usize_argument(&arguments, 3, 100, "batch size")?;
    let hidden_size = parse_usize_argument(&arguments, 4, 128, "hidden size")?;
    let learning_rate = parse_f32_argument(&arguments, 5, 0.001, "learning rate")?;

    println!("Loading MNIST from {}", data_directory.display());

    let train_dataset = MnistDataset::load(
        data_directory.join("train-images-idx3-ubyte"),
        data_directory.join("train-labels-idx1-ubyte"),
    )?;
    let test_dataset = MnistDataset::load(
        data_directory.join("t10k-images-idx3-ubyte"),
        data_directory.join("t10k-labels-idx1-ubyte"),
    )?;

    println!(
        "Training images: {} | Test images: {} | Image size: {}x{}",
        train_dataset.len(),
        test_dataset.len(),
        train_dataset.rows(),
        train_dataset.columns()
    );

    let mut rng = Rng::new(12345);
    let mut network = TwoLayerNetwork::new(train_dataset.image_size(), hidden_size, 10, &mut rng);
    let optimizer = Adam::standard(learning_rate);
    let mut optimizer_state = network.adam_state();

    println!(
        "Network: {} -> {} -> 10 | Batch size: {} | Learning rate: {}",
        train_dataset.image_size(),
        hidden_size,
        batch_size,
        learning_rate
    );

    for epoch in 1..=epochs {
        let start = Instant::now();
        let training_loss = train_epoch_adam(
            &mut network,
            &train_dataset,
            batch_size,
            &optimizer,
            &mut optimizer_state,
            &mut rng,
        );
        let test_metrics = evaluate(&mut network, &test_dataset, batch_size);
        let elapsed = start.elapsed();

        println!(
            "Epoch {:2}/{} | train loss {:.5} | test loss {:.5} | test accuracy {:6.2}% | {:.2?}",
            epoch,
            epochs,
            training_loss,
            test_metrics.loss,
            test_metrics.accuracy * 100.0,
            elapsed
        );

        network.save("mnist_model.bin")?;
    }

    println!("Saved trained parameters to mnist_model.bin");
    Ok(())
}

fn parse_usize_argument(
    arguments: &[String],
    index: usize,
    default: usize,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    match arguments.get(index) {
        Some(value) => {
            let parsed: usize = value.parse()?;
            if parsed == 0 {
                return Err(format!("{name} must be greater than zero").into());
            }
            Ok(parsed)
        }
        None => Ok(default),
    }
}

fn parse_f32_argument(
    arguments: &[String],
    index: usize,
    default: f32,
    name: &str,
) -> Result<f32, Box<dyn Error>> {
    match arguments.get(index) {
        Some(value) => {
            let parsed: f32 = value.parse()?;
            if parsed <= 0.0 {
                return Err(format!("{name} must be greater than zero").into());
            }
            Ok(parsed)
        }
        None => Ok(default),
    }
}
