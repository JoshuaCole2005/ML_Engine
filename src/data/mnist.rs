use std::fs;
use std::io;
use std::path::Path;

use crate::Tensor;

const IMAGE_MAGIC: u32 = 2051;
const LABEL_MAGIC: u32 = 2049;

#[derive(Clone, Debug)]
pub struct MnistDataset {
    images: Vec<f32>,
    labels: Vec<usize>,
    rows: usize,
    columns: usize,
}

impl MnistDataset {
    pub fn load<I: AsRef<Path>, L: AsRef<Path>>(image_path: I, label_path: L) -> io::Result<Self> {
        let image_bytes = fs::read(image_path)?;
        let label_bytes = fs::read(label_path)?;
        Self::from_bytes(&image_bytes, &label_bytes)
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn image_size(&self) -> usize {
        self.rows * self.columns
    }

    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    pub fn image(&self, index: usize) -> &[f32] {
        assert!(index < self.len(), "MNIST image index is out of bounds");
        let image_size = self.image_size();
        let start = index * image_size;
        &self.images[start..start + image_size]
    }

    pub fn batch(&self, indices: &[usize]) -> (Tensor, Vec<usize>) {
        assert!(!indices.is_empty(), "MNIST batch cannot be empty");
        let mut inputs = Tensor::zeros(&[indices.len(), self.image_size()]);
        let mut labels = Vec::with_capacity(indices.len());
        self.fill_batch(indices, &mut inputs, &mut labels);
        (inputs, labels)
    }

    pub fn batch_range(&self, start: usize, end: usize) -> (Tensor, Vec<usize>) {
        assert!(start < end, "MNIST batch range cannot be empty");
        assert!(
            end <= self.len(),
            "MNIST batch range exceeds dataset length"
        );
        let indices: Vec<usize> = (start..end).collect();
        self.batch(&indices)
    }

    pub fn fill_batch(&self, indices: &[usize], inputs: &mut Tensor, labels: &mut Vec<usize>) {
        assert!(!indices.is_empty(), "MNIST batch cannot be empty");
        assert_eq!(
            inputs.shape(),
            &[indices.len(), self.image_size()],
            "batch input tensor has shape {:?}, but expected [{}, {}]",
            inputs.shape(),
            indices.len(),
            self.image_size()
        );

        labels.clear();
        if labels.capacity() < indices.len() {
            labels.reserve(indices.len());
        }

        let image_size = self.image_size();
        for (batch_row, dataset_index) in indices.iter().copied().enumerate() {
            assert!(
                dataset_index < self.len(),
                "MNIST batch index is out of bounds"
            );
            let source_start = dataset_index * image_size;
            let destination_start = batch_row * image_size;
            inputs.data_mut()[destination_start..destination_start + image_size]
                .copy_from_slice(&self.images[source_start..source_start + image_size]);
            labels.push(self.labels[dataset_index]);
        }
    }

    fn from_bytes(image_bytes: &[u8], label_bytes: &[u8]) -> io::Result<Self> {
        if image_bytes.len() < 16 {
            return Err(invalid_data("MNIST image file is shorter than its header"));
        }
        if label_bytes.len() < 8 {
            return Err(invalid_data("MNIST label file is shorter than its header"));
        }

        let image_magic = read_be_u32(image_bytes, 0)?;
        let image_count = read_be_u32(image_bytes, 4)? as usize;
        let rows = read_be_u32(image_bytes, 8)? as usize;
        let columns = read_be_u32(image_bytes, 12)? as usize;

        if image_magic != IMAGE_MAGIC {
            return Err(invalid_data(&format!(
                "MNIST image magic number was {image_magic}, expected {IMAGE_MAGIC}"
            )));
        }
        if image_count == 0 || rows == 0 || columns == 0 {
            return Err(invalid_data(
                "MNIST image dimensions must be greater than zero",
            ));
        }

        let image_size = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid_data("MNIST image dimensions are too large"))?;
        let pixel_count = image_count
            .checked_mul(image_size)
            .ok_or_else(|| invalid_data("MNIST image file is too large"))?;
        let expected_image_length = 16_usize
            .checked_add(pixel_count)
            .ok_or_else(|| invalid_data("MNIST image file length overflowed"))?;

        if image_bytes.len() != expected_image_length {
            return Err(invalid_data(&format!(
                "MNIST image file has {} bytes, expected {}",
                image_bytes.len(),
                expected_image_length
            )));
        }

        let label_magic = read_be_u32(label_bytes, 0)?;
        let label_count = read_be_u32(label_bytes, 4)? as usize;

        if label_magic != LABEL_MAGIC {
            return Err(invalid_data(&format!(
                "MNIST label magic number was {label_magic}, expected {LABEL_MAGIC}"
            )));
        }
        if label_count != image_count {
            return Err(invalid_data(&format!(
                "MNIST contains {image_count} images but {label_count} labels"
            )));
        }

        let expected_label_length = 8_usize
            .checked_add(label_count)
            .ok_or_else(|| invalid_data("MNIST label file length overflowed"))?;
        if label_bytes.len() != expected_label_length {
            return Err(invalid_data(&format!(
                "MNIST label file has {} bytes, expected {}",
                label_bytes.len(),
                expected_label_length
            )));
        }

        let images = image_bytes[16..]
            .iter()
            .map(|pixel| *pixel as f32 / 255.0)
            .collect();

        let mut labels = Vec::with_capacity(label_count);
        for label in &label_bytes[8..] {
            if *label > 9 {
                return Err(invalid_data(&format!(
                    "MNIST label {} is outside the digit range 0..9",
                    label
                )));
            }
            labels.push(*label as usize);
        }

        Ok(Self {
            images,
            labels,
            rows,
            columns,
        })
    }
}

fn read_be_u32(bytes: &[u8], offset: usize) -> io::Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| invalid_data("IDX header offset overflowed"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_data("IDX header ended unexpectedly"))?;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::MnistDataset;
    use crate::Tensor;

    fn synthetic_files() -> (Vec<u8>, Vec<u8>) {
        let mut images = Vec::new();
        images.extend_from_slice(&2051_u32.to_be_bytes());
        images.extend_from_slice(&2_u32.to_be_bytes());
        images.extend_from_slice(&2_u32.to_be_bytes());
        images.extend_from_slice(&2_u32.to_be_bytes());
        images.extend_from_slice(&[0, 255, 128, 64, 10, 20, 30, 40]);

        let mut labels = Vec::new();
        labels.extend_from_slice(&2049_u32.to_be_bytes());
        labels.extend_from_slice(&2_u32.to_be_bytes());
        labels.extend_from_slice(&[3, 7]);
        (images, labels)
    }

    #[test]
    fn parses_images_labels_and_normalizes_pixels() {
        let (images, labels) = synthetic_files();
        let dataset = MnistDataset::from_bytes(&images, &labels).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.rows(), 2);
        assert_eq!(dataset.columns(), 2);
        assert_eq!(dataset.labels(), &[3, 7]);
        assert_eq!(dataset.image(0)[0], 0.0);
        assert_eq!(dataset.image(0)[1], 1.0);
        assert!((dataset.image(0)[2] - 128.0 / 255.0).abs() < 0.000001);
    }

    #[test]
    fn fills_reusable_batch_buffer_in_requested_order() {
        let (images, labels) = synthetic_files();
        let dataset = MnistDataset::from_bytes(&images, &labels).unwrap();
        let mut inputs = Tensor::zeros(&[2, 4]);
        let mut batch_labels = Vec::new();

        dataset.fill_batch(&[1, 0], &mut inputs, &mut batch_labels);

        assert_eq!(batch_labels, vec![7, 3]);
        assert!((inputs.data()[0] - 10.0 / 255.0).abs() < 0.000001);
        assert_eq!(inputs.data()[5], 1.0);
    }

    #[test]
    fn rejects_wrong_magic_number() {
        let (mut images, labels) = synthetic_files();
        images[3] = 0;
        assert!(MnistDataset::from_bytes(&images, &labels).is_err());
    }
}
