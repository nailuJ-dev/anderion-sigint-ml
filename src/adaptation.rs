use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanVarianceAdapter {
    source_mean: Vec<f32>,
    source_std: Vec<f32>,
    target_mean: Vec<f32>,
    target_std: Vec<f32>,
    epsilon: f32,
}

impl MeanVarianceAdapter {
    pub fn fit(source: &[Embedding], target: &[Embedding]) -> Result<Self> {
        if source.is_empty() || target.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let dim = source[0].dim();
        if source
            .iter()
            .chain(target)
            .any(|sample| sample.dim() != dim)
        {
            return Err(SdkError::InvalidArgument(
                "domain samples must share a dimension".into(),
            ));
        }
        let (source_mean, source_std) = moments(source, dim);
        let (target_mean, target_std) = moments(target, dim);
        Ok(Self {
            source_mean,
            source_std,
            target_mean,
            target_std,
            epsilon: 1e-6,
        })
    }

    pub fn transform(&self, embedding: &Embedding) -> Result<Embedding> {
        if embedding.dim() != self.source_mean.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.source_mean.len(),
                actual: embedding.dim(),
            });
        }
        let values = embedding
            .values()
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let z = (*value - self.source_mean[i]) / (self.source_std[i] + self.epsilon);
                z * self.target_std[i] + self.target_mean[i]
            })
            .collect();
        Embedding::new(values)
    }
}

fn moments(samples: &[Embedding], dim: usize) -> (Vec<f32>, Vec<f32>) {
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
    let mut var = vec![0.0_f32; dim];
    for sample in samples {
        for ((dst, src), avg) in var.iter_mut().zip(sample.values()).zip(&mean) {
            let delta = *src - *avg;
            *dst += delta * delta;
        }
    }
    for value in &mut var {
        *value = (*value / n).sqrt();
    }
    (mean, var)
}
