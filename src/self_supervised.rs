use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedContextPretrainer {
    dimension: usize,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl MaskedContextPretrainer {
    pub fn fit(samples: &[Embedding], epochs: usize, learning_rate: f32, l2: f32) -> Result<Self> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if epochs == 0 {
            return Err(SdkError::InvalidArgument("epochs must be positive".into()));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "learning_rate must be finite and positive".into(),
            ));
        }
        if !l2.is_finite() || l2 < 0.0 {
            return Err(SdkError::InvalidArgument(
                "l2 must be finite and non-negative".into(),
            ));
        }
        let dimension = samples[0].dim();
        if dimension > 512 {
            return Err(SdkError::DimensionLimit {
                actual: dimension,
                max: 512,
            });
        }
        if samples.iter().any(|sample| sample.dim() != dimension) {
            return Err(SdkError::InvalidArgument(
                "self-supervised samples must share a dimension".into(),
            ));
        }
        let mut bias = vec![0.0_f32; dimension];
        for sample in samples {
            for (dst, src) in bias.iter_mut().zip(sample.values()) {
                *dst += *src;
            }
        }
        for value in &mut bias {
            *value /= samples.len() as f32;
        }
        let mut weights = vec![vec![0.0_f32; dimension]; dimension];
        for _ in 0..epochs {
            for sample in samples {
                for target in 0..dimension {
                    let mut prediction = bias[target];
                    for (source, weight) in weights[target].iter().enumerate() {
                        if source != target {
                            prediction += *weight * sample.values()[source];
                        }
                    }
                    let error = prediction - sample.values()[target];
                    bias[target] -= learning_rate * 2.0 * error;
                    for (source, weight) in weights[target].iter_mut().enumerate() {
                        if source == target {
                            continue;
                        }
                        let gradient = 2.0 * error * sample.values()[source] + 2.0 * l2 * *weight;
                        *weight -= learning_rate * gradient;
                    }
                }
            }
        }
        Ok(Self {
            dimension,
            weights,
            bias,
        })
    }

    pub fn reconstruct(&self, embedding: &Embedding, masked_dimension: usize) -> Result<f32> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        if masked_dimension >= self.dimension {
            return Err(SdkError::InvalidArgument(
                "masked dimension is out of range".into(),
            ));
        }
        let mut value = self.bias[masked_dimension];
        for source in 0..self.dimension {
            if source != masked_dimension {
                value += self.weights[masked_dimension][source] * embedding.values()[source];
            }
        }
        if !value.is_finite() {
            return Err(SdkError::InvalidArgument(
                "reconstruction became non-finite".into(),
            ));
        }
        Ok(value)
    }

    pub fn reconstruction_errors(&self, embedding: &Embedding) -> Result<Embedding> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        let mut errors = Vec::with_capacity(self.dimension);
        for target in 0..self.dimension {
            errors.push((embedding.values()[target] - self.reconstruct(embedding, target)?).abs());
        }
        Embedding::new(errors)
    }
}
