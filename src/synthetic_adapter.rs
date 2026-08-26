use crate::{Observation, Result, SdkError};

pub trait SyntheticDataAdapter {
    fn generate(&self, count: usize) -> Result<Vec<Observation>>;
}

#[derive(Debug, Clone, Copy)]
pub struct SyntheticFeatureGenerator {
    dimension: usize,
    seed: u64,
}

impl SyntheticFeatureGenerator {
    pub fn new(dimension: usize, seed: u64) -> Result<Self> {
        if dimension == 0 || dimension > 65_536 {
            return Err(SdkError::DimensionLimit {
                actual: dimension,
                max: 65_536,
            });
        }
        Ok(Self { dimension, seed })
    }
}

impl SyntheticDataAdapter for SyntheticFeatureGenerator {
    fn generate(&self, count: usize) -> Result<Vec<Observation>> {
        if count == 0 || count > 65_536 {
            return Err(SdkError::InvalidArgument(
                "synthetic count must be in 1..=65536".into(),
            ));
        }
        let elements = count
            .checked_mul(self.dimension)
            .ok_or_else(|| SdkError::InvalidArgument("synthetic element count overflow".into()))?;
        if elements > 4_194_304 {
            return Err(SdkError::DimensionLimit {
                actual: elements,
                max: 4_194_304,
            });
        }
        let mut output = Vec::with_capacity(count);
        for row in 0..count {
            let mut features = Vec::with_capacity(self.dimension);
            for column in 0..self.dimension {
                features.push(pseudo_feature(self.seed, row, column));
            }
            output.push(Observation::new(
                format!("synthetic-{row}"),
                row as u64,
                features,
            )?);
        }
        Ok(output)
    }
}

fn pseudo_feature(seed: u64, row: usize, column: usize) -> f32 {
    let mut x = seed
        ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (column as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = x as f64 / u64::MAX as f64;
    (unit.mul_add(2.0, -1.0)) as f32
}
