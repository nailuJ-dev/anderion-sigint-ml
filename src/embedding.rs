use serde::{Deserialize, Serialize};

use crate::{Embedding, Encoder, Observation, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashProjectionEncoder {
    input_dim: usize,
    output_dim: usize,
    seed: u64,
}

impl HashProjectionEncoder {
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(SdkError::InvalidArgument(
                "encoder dimensions must be positive".into(),
            ));
        }
        if input_dim > 65_536 || output_dim > 8_192 {
            return Err(SdkError::DimensionLimit {
                actual: input_dim.max(output_dim),
                max: 65_536,
            });
        }
        Ok(Self {
            input_dim,
            output_dim,
            seed,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.input_dim == 0 || self.output_dim == 0 {
            return Err(SdkError::InvalidArgument(
                "encoder dimensions must be positive".into(),
            ));
        }
        if self.input_dim > 65_536 || self.output_dim > 8_192 {
            return Err(SdkError::DimensionLimit {
                actual: self.input_dim.max(self.output_dim),
                max: 65_536,
            });
        }
        Ok(())
    }

    pub fn encode_features(&self, features: &[f32]) -> Result<Embedding> {
        if features.len() != self.input_dim {
            return Err(SdkError::DimensionMismatch {
                expected: self.input_dim,
                actual: features.len(),
            });
        }
        if let Some((index, _)) = features
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(SdkError::NonFiniteValue { index });
        }
        let scale = (self.input_dim as f32).sqrt().max(1.0);
        let mut output = vec![0.0_f32; self.output_dim];
        for (out_idx, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for (in_idx, value) in features.iter().copied().enumerate() {
                sum += value * pseudo_weight(self.seed, out_idx, in_idx);
            }
            *out = (sum / scale).tanh();
        }
        Embedding::new(output)
    }
}

impl Encoder for HashProjectionEncoder {
    fn encode(&self, observation: &Observation) -> Result<Embedding> {
        self.encode_features(observation.features())
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }
    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

fn pseudo_weight(seed: u64, output: usize, input: usize) -> f32 {
    let mut x = seed
        ^ (output as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (input as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = (x as f64) / (u64::MAX as f64);
    (unit.mul_add(2.0, -1.0)) as f32
}
