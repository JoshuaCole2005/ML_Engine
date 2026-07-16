#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let element_count = Self::element_count(shape);
        let strides = Self::calculate_strides(shape);
        Self {
            data: vec![0.0; element_count],
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn filled(shape: &[usize], value: f32) -> Self {
        let element_count = Self::element_count(shape);
        let strides = Self::calculate_strides(shape);
        Self {
            data: vec![value; element_count],
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let expected_count = Self::element_count(shape);
        assert_eq!(
            data.len(),
            expected_count,
            "data length {} does not match shape {:?}, which requires {} values",
            data.len(),
            shape,
            expected_count
        );
        let strides = Self::calculate_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        let flat_index = self.flat_index(indices);
        self.data[flat_index]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let flat_index = self.flat_index(indices);
        self.data[flat_index] = value;
    }

    pub fn fill(&mut self, value: f32) {
        for current_value in &mut self.data {
            *current_value = value;
        }
    }

    pub fn copy_from(&mut self, other: &Tensor) {
        self.assert_same_shape(other, "tensor copy");
        self.data.copy_from_slice(other.data());
    }

    pub fn reshape(&mut self, new_shape: &[usize]) {
        let new_element_count = Self::element_count(new_shape);
        assert_eq!(
            self.len(),
            new_element_count,
            "cannot reshape tensor with {} values into shape {:?}, which requires {} values",
            self.len(),
            new_shape,
            new_element_count
        );
        self.shape = new_shape.to_vec();
        self.strides = Self::calculate_strides(new_shape);
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.assert_same_shape(other, "elementwise addition");
        let mut result = Tensor::zeros(self.shape());
        for index in 0..self.len() {
            result.data[index] = self.data[index] + other.data[index];
        }
        result
    }

    pub fn add_in_place(&mut self, other: &Tensor) {
        self.assert_same_shape(other, "in-place elementwise addition");
        for index in 0..self.len() {
            self.data[index] += other.data[index];
        }
    }

    pub fn subtract(&self, other: &Tensor) -> Tensor {
        self.assert_same_shape(other, "elementwise subtraction");
        let mut result = Tensor::zeros(self.shape());
        for index in 0..self.len() {
            result.data[index] = self.data[index] - other.data[index];
        }
        result
    }

    pub fn subtract_in_place(&mut self, other: &Tensor) {
        self.assert_same_shape(other, "in-place elementwise subtraction");
        for index in 0..self.len() {
            self.data[index] -= other.data[index];
        }
    }

    pub fn multiply_elementwise(&self, other: &Tensor) -> Tensor {
        self.assert_same_shape(other, "elementwise multiplication");
        let mut result = Tensor::zeros(self.shape());
        for index in 0..self.len() {
            result.data[index] = self.data[index] * other.data[index];
        }
        result
    }

    pub fn multiply_scalar(&self, scalar: f32) -> Tensor {
        let mut result = Tensor::zeros(self.shape());
        for index in 0..self.len() {
            result.data[index] = self.data[index] * scalar;
        }
        result
    }

    pub fn multiply_scalar_in_place(&mut self, scalar: f32) {
        for value in &mut self.data {
            *value *= scalar;
        }
    }

    pub fn add_row_vector(&self, row_vector: &Tensor) -> Tensor {
        self.assert_rank(2, "row-vector broadcasting");
        row_vector.assert_rank(1, "row-vector broadcasting");
        let rows = self.shape[0];
        let columns = self.shape[1];
        assert_eq!(
            row_vector.shape[0], columns,
            "row vector has length {}, but matrix has {} columns",
            row_vector.shape[0], columns
        );
        let mut result = Tensor::zeros(self.shape());
        for row in 0..rows {
            let row_start = row * columns;
            for column in 0..columns {
                let index = row_start + column;
                result.data[index] = self.data[index] + row_vector.data[column];
            }
        }
        result
    }

    pub fn add_row_vector_in_place(&mut self, row_vector: &Tensor) {
        self.assert_rank(2, "in-place row-vector broadcasting");
        row_vector.assert_rank(1, "in-place row-vector broadcasting");
        let rows = self.shape[0];
        let columns = self.shape[1];
        assert_eq!(
            row_vector.shape[0], columns,
            "row vector has length {}, but matrix has {} columns",
            row_vector.shape[0], columns
        );
        for row in 0..rows {
            let row_start = row * columns;
            for column in 0..columns {
                self.data[row_start + column] += row_vector.data[column];
            }
        }
    }

    pub fn transpose_2d(&self) -> Tensor {
        self.assert_rank(2, "2D transpose");
        let rows = self.shape[0];
        let columns = self.shape[1];
        let mut result = Tensor::zeros(&[columns, rows]);
        for row in 0..rows {
            for column in 0..columns {
                let source_index = row * columns + column;
                let destination_index = column * rows + row;
                result.data[destination_index] = self.data[source_index];
            }
        }
        result
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        self.assert_rank(2, "matrix multiplication");
        other.assert_rank(2, "matrix multiplication");
        let left_rows = self.shape[0];
        let left_columns = self.shape[1];
        let right_rows = other.shape[0];
        let right_columns = other.shape[1];
        assert_eq!(
            left_columns,
            right_rows,
            "matrix multiplication cannot multiply shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        let mut output = Tensor::zeros(&[left_rows, right_columns]);
        self.matmul_into(other, &mut output);
        output
    }

    pub fn matmul_into(&self, other: &Tensor, output: &mut Tensor) {
        self.matmul_into_blocked(other, output, 32);
    }

    pub fn matmul_into_blocked(&self, other: &Tensor, output: &mut Tensor, block_size: usize) {
        self.assert_rank(2, "matrix multiplication");
        other.assert_rank(2, "matrix multiplication");
        output.assert_rank(2, "matrix multiplication output");
        assert!(
            block_size > 0,
            "matrix multiplication block size must be positive"
        );

        let left_rows = self.shape[0];
        let left_columns = self.shape[1];
        let right_rows = other.shape[0];
        let right_columns = other.shape[1];

        assert_eq!(
            left_columns,
            right_rows,
            "matrix multiplication cannot multiply shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        assert_eq!(
            output.shape(),
            &[left_rows, right_columns],
            "matrix multiplication output has shape {:?}, but expected [{}, {}]",
            output.shape(),
            left_rows,
            right_columns
        );

        output.fill(0.0);

        for row_block in (0..left_rows).step_by(block_size) {
            let row_end = (row_block + block_size).min(left_rows);

            for shared_block in (0..left_columns).step_by(block_size) {
                let shared_end = (shared_block + block_size).min(left_columns);

                for column_block in (0..right_columns).step_by(block_size) {
                    let column_end = (column_block + block_size).min(right_columns);

                    for left_row in row_block..row_end {
                        let left_row_start = left_row * left_columns;
                        let output_row_start = left_row * right_columns;

                        for shared_index in shared_block..shared_end {
                            let left_value = self.data[left_row_start + shared_index];
                            let right_row_start = shared_index * right_columns;

                            for right_column in column_block..column_end {
                                output.data[output_row_start + right_column] +=
                                    left_value * other.data[right_row_start + right_column];
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn sum(&self) -> f32 {
        let mut total = 0.0;
        for value in &self.data {
            total += *value;
        }
        total
    }

    pub fn sum_rows(&self) -> Tensor {
        self.assert_rank(2, "row summation");
        let columns = self.shape[1];
        let mut result = Tensor::zeros(&[columns]);
        self.sum_rows_into(&mut result);
        result
    }

    pub fn sum_rows_into(&self, output: &mut Tensor) {
        self.assert_rank(2, "row summation");
        output.assert_rank(1, "row summation output");
        let rows = self.shape[0];
        let columns = self.shape[1];
        assert_eq!(
            output.shape(),
            &[columns],
            "row summation output has shape {:?}, but expected [{}]",
            output.shape(),
            columns
        );
        output.fill(0.0);
        for row in 0..rows {
            let row_start = row * columns;
            for column in 0..columns {
                output.data[column] += self.data[row_start + column];
            }
        }
    }

    pub fn argmax_rows(&self) -> Vec<usize> {
        self.assert_rank(2, "row-wise argmax");
        let rows = self.shape[0];
        let columns = self.shape[1];
        let mut result = vec![0; rows];
        for row in 0..rows {
            let row_start = row * columns;
            let mut best_column = 0;
            let mut best_value = self.data[row_start];
            for column in 1..columns {
                let current_value = self.data[row_start + column];
                if current_value > best_value {
                    best_value = current_value;
                    best_column = column;
                }
            }
            result[row] = best_column;
        }
        result
    }

    fn element_count(shape: &[usize]) -> usize {
        assert!(
            !shape.is_empty(),
            "a tensor must have at least one dimension"
        );
        let mut count: usize = 1;
        for &dimension_size in shape {
            assert!(
                dimension_size > 0,
                "tensor dimensions must be greater than zero"
            );
            count = count
                .checked_mul(dimension_size)
                .expect("tensor is too large");
        }
        count
    }

    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        if shape.len() > 1 {
            for dimension in (0..shape.len() - 1).rev() {
                strides[dimension] = strides[dimension + 1] * shape[dimension + 1];
            }
        }
        strides
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "received {} indices for a tensor with {} dimensions",
            indices.len(),
            self.shape.len()
        );
        let mut flat_index = 0;
        for dimension in 0..indices.len() {
            let index = indices[dimension];
            let dimension_size = self.shape[dimension];
            assert!(
                index < dimension_size,
                "index {} is outside dimension {} with size {}",
                index,
                dimension,
                dimension_size
            );
            flat_index += index * self.strides[dimension];
        }
        flat_index
    }

    fn assert_same_shape(&self, other: &Tensor, operation: &str) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "{} requires matching shapes, but received {:?} and {:?}",
            operation,
            self.shape(),
            other.shape()
        );
    }

    fn assert_rank(&self, expected_rank: usize, operation: &str) {
        assert_eq!(
            self.rank(),
            expected_rank,
            "{} requires a rank-{} tensor, but received shape {:?}",
            operation,
            expected_rank,
            self.shape()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn blocked_matrix_multiplication_matches_expected_values() {
        let left = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let right = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let mut output = Tensor::zeros(&[2, 2]);
        left.matmul_into_blocked(&right, &mut output, 1);
        assert_eq!(output.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn sum_rows_into_reuses_output() {
        let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut output = Tensor::filled(&[3], 99.0);
        matrix.sum_rows_into(&mut output);
        assert_eq!(output.data(), &[5.0, 7.0, 9.0]);
    }
}
