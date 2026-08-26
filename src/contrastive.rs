use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveProjector {
    scales: Vec<f32>,
}

impl ContrastiveProjector {
    pub fn fit(pairs: &[(Embedding, Embedding, bool)]) -> Result<Self> {
        if pairs.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let dim = pairs[0].0.dim();
        if pairs
            .iter()
            .any(|(a, b, _)| a.dim() != dim || b.dim() != dim)
        {
            return Err(SdkError::InvalidArgument(
                "contrastive pair dimensions must match".into(),
            ));
        }
        let mut positive = vec![0.0_f32; dim];
        let mut negative = vec![0.0_f32; dim];
        let mut positive_count = 0_u64;
        let mut negative_count = 0_u64;
        for (a, b, same) in pairs {
            let target = if *same { &mut positive } else { &mut negative };
            if *same {
                positive_count = positive_count.saturating_add(1);
            } else {
                negative_count = negative_count.saturating_add(1);
            }
            for ((dst, x), y) in target.iter_mut().zip(a.values()).zip(b.values()) {
                let delta = x - y;
                *dst += delta * delta;
            }
        }
        if positive_count == 0 || negative_count == 0 {
            return Err(SdkError::InvalidArgument(
                "contrastive fitting requires positive and negative pairs".into(),
            ));
        }
        let mut scales = Vec::with_capacity(dim);
        for index in 0..dim {
            let pos = positive[index] / positive_count as f32;
            let neg = negative[index] / negative_count as f32;
            let scale = ((neg + 1.0e-6) / (pos + 1.0e-6)).sqrt().clamp(0.05, 20.0);
            scales.push(scale);
        }
        let model = Self { scales };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<()> {
        if self.scales.is_empty()
            || self.scales.len() > 8_192
            || self
                .scales
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(SdkError::InvalidArgument(
                "invalid contrastive projector scales".into(),
            ));
        }
        Ok(())
    }

    pub fn transform(&self, embedding: &Embedding) -> Result<Embedding> {
        self.validate()?;
        if embedding.dim() != self.scales.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.scales.len(),
                actual: embedding.dim(),
            });
        }
        let mut values: Vec<f32> = embedding
            .values()
            .iter()
            .zip(&self.scales)
            .map(|(value, scale)| value * scale)
            .collect();
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for value in &mut values {
                *value /= norm;
            }
        }
        Embedding::new(values)
    }
}
