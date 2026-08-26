use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Copy)]
pub struct TemporalSelfAttentionEncoder {
    temperature: f32,
}

impl TemporalSelfAttentionEncoder {
    pub fn new(temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "attention temperature must be finite and positive".into(),
            ));
        }
        Ok(Self { temperature })
    }

    pub fn aggregate(&self, sequence: &[Embedding]) -> Result<Embedding> {
        if sequence.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if sequence.len() > 65_536 {
            return Err(SdkError::DimensionLimit {
                actual: sequence.len(),
                max: 65_536,
            });
        }
        let dim = sequence[0].dim();
        if sequence.iter().any(|item| item.dim() != dim) {
            let actual = sequence.iter().map(Embedding::dim).max().unwrap_or(dim);
            return Err(SdkError::DimensionMismatch {
                expected: dim,
                actual,
            });
        }
        let query = sequence.last().ok_or(SdkError::EmptyDataset)?;
        let normalization = (dim as f32).sqrt().max(1.0) * self.temperature;
        let logits: Vec<f32> = sequence
            .iter()
            .map(|key| dot(query.values(), key.values()) / normalization)
            .collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|value| (*value - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "attention normalization failed".into(),
            ));
        }
        let mut pooled = vec![0.0_f32; dim];
        for (item, weight) in sequence.iter().zip(exp.iter().map(|value| *value / sum)) {
            for (dst, value) in pooled.iter_mut().zip(item.values()) {
                *dst += weight * *value;
            }
        }
        Embedding::new(pooled)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalTransformerEncoder {
    attention: TemporalSelfAttentionEncoder,
    residual_weight: f32,
}

impl TemporalTransformerEncoder {
    pub fn new(temperature: f32, residual_weight: f32) -> Result<Self> {
        if !residual_weight.is_finite() || !(0.0..=1.0).contains(&residual_weight) {
            return Err(SdkError::InvalidProbability(residual_weight));
        }
        Ok(Self {
            attention: TemporalSelfAttentionEncoder::new(temperature)?,
            residual_weight,
        })
    }

    pub fn aggregate(&self, sequence: &[Embedding]) -> Result<Embedding> {
        let attended = self.attention.aggregate(sequence)?;
        let residual = sequence.last().ok_or(SdkError::EmptyDataset)?;
        if residual.dim() != attended.dim() {
            return Err(SdkError::DimensionMismatch {
                expected: attended.dim(),
                actual: residual.dim(),
            });
        }
        let mixed: Vec<f32> = attended
            .values()
            .iter()
            .zip(residual.values())
            .map(|(attention, last)| {
                ((1.0 - self.residual_weight) * *attention + self.residual_weight * *last).tanh()
            })
            .collect();
        Embedding::new(layer_normalize(mixed))
    }
}

fn layer_normalize(mut values: Vec<f32>) -> Vec<f32> {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    let denom = (variance + 1.0e-5).sqrt();
    for value in &mut values {
        *value = (*value - mean) / denom;
    }
    values
}
