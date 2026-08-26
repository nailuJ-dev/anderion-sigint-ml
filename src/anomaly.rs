use serde::{Deserialize, Serialize};

use crate::{AnomalyDetector, Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagonalGaussianAnomalyDetector {
    mean: Vec<f32>,
    variance: Vec<f32>,
    epsilon: f32,
}

impl DiagonalGaussianAnomalyDetector {
    pub fn fit(samples: &[Embedding]) -> Result<Self> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let dim = samples[0].dim();
        if samples.iter().any(|sample| sample.dim() != dim) {
            return Err(SdkError::InvalidArgument(
                "all anomaly samples must share a dimension".into(),
            ));
        }
        let mut mean = vec![0.0_f32; dim];
        for sample in samples {
            for (dst, src) in mean.iter_mut().zip(sample.values()) {
                *dst += *src;
            }
        }
        let n = samples.len() as f32;
        for value in &mut mean {
            *value /= n;
        }

        let mut variance = vec![0.0_f32; dim];
        for sample in samples {
            for ((var, value), avg) in variance.iter_mut().zip(sample.values()).zip(&mean) {
                let delta = *value - *avg;
                *var += delta * delta;
            }
        }
        for value in &mut variance {
            *value /= n.max(1.0);
        }
        Ok(Self {
            mean,
            variance,
            epsilon: 1e-6,
        })
    }
}

impl AnomalyDetector for DiagonalGaussianAnomalyDetector {
    fn anomaly_score(&self, embedding: &Embedding) -> Result<f32> {
        if embedding.dim() != self.mean.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.mean.len(),
                actual: embedding.dim(),
            });
        }
        let raw: f32 = embedding
            .values()
            .iter()
            .zip(&self.mean)
            .zip(&self.variance)
            .map(|((value, mean), variance)| {
                let delta = *value - *mean;
                (delta * delta) / (*variance + self.epsilon)
            })
            .sum::<f32>()
            / self.mean.len() as f32;
        Ok(1.0 - (-0.5 * raw.max(0.0)).exp())
    }

    fn input_dim(&self) -> usize {
        self.mean.len()
    }
}
