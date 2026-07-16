#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / 16_777_216.0
    }

    pub fn uniform(&mut self, minimum: f32, maximum: f32) -> f32 {
        assert!(minimum < maximum, "minimum must be smaller than maximum");
        minimum + (maximum - minimum) * self.next_f32()
    }

    pub fn normal(&mut self, mean: f32, standard_deviation: f32) -> f32 {
        assert!(
            standard_deviation >= 0.0,
            "standard deviation cannot be negative"
        );
        let first_uniform = 1.0 - self.next_f32();
        let second_uniform = self.next_f32();
        let radius = (-2.0 * first_uniform.ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * second_uniform;
        mean + standard_deviation * radius * angle.cos()
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        if values.len() < 2 {
            return;
        }
        for current_index in (1..values.len()).rev() {
            let random_index = (self.next_u64() % (current_index as u64 + 1)) as usize;
            values.swap(current_index, random_index);
        }
    }
}
