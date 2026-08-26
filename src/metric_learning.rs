use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagonalMetricLearner {
    weights: Vec<f32>,
}

impl DiagonalMetricLearner {
    pub fn fit(
        positive_pairs: &[(Embedding, Embedding)],
        negative_pairs: &[(Embedding, Embedding)],
        regularization: f32,
    ) -> Result<Self> {
        if positive_pairs.is_empty() || negative_pairs.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if !regularization.is_finite() || regularization <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "regularization must be finite and positive".into(),
            ));
        }
        let dim = positive_pairs[0].0.dim();
        let all = positive_pairs.iter().chain(negative_pairs);
        for (a, b) in all {
            if a.dim() != dim || b.dim() != dim {
                return Err(SdkError::DimensionMismatch {
                    expected: dim,
                    actual: a.dim().max(b.dim()),
                });
            }
        }
        let pos = mean_squared_delta(positive_pairs, dim);
        let neg = mean_squared_delta(negative_pairs, dim);
        let mut weights: Vec<f32> = neg
            .iter()
            .zip(&pos)
            .map(|(n, p)| ((n + regularization) / (p + regularization)).sqrt())
            .collect();
        let mean_weight = weights.iter().sum::<f32>() / weights.len() as f32;
        if mean_weight > 0.0 {
            for weight in &mut weights {
                *weight /= mean_weight;
            }
        }
        Ok(Self { weights })
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn transform(&self, embedding: &Embedding) -> Result<Embedding> {
        if embedding.dim() != self.weights.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.weights.len(),
                actual: embedding.dim(),
            });
        }
        Embedding::new(
            embedding
                .values()
                .iter()
                .zip(&self.weights)
                .map(|(x, w)| x * w)
                .collect(),
        )
    }

    pub fn distance(&self, a: &Embedding, b: &Embedding) -> Result<f32> {
        if a.dim() != self.weights.len() || b.dim() != self.weights.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.weights.len(),
                actual: a.dim().max(b.dim()),
            });
        }
        Ok(a.values()
            .iter()
            .zip(b.values())
            .zip(&self.weights)
            .map(|((x, y), w)| {
                let d = (x - y) * w;
                d * d
            })
            .sum::<f32>()
            .sqrt())
    }
}

fn mean_squared_delta(pairs: &[(Embedding, Embedding)], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; dim];
    for (a, b) in pairs {
        for ((dst, x), y) in out.iter_mut().zip(a.values()).zip(b.values()) {
            let delta = *x - *y;
            *dst += delta * delta;
        }
    }
    let n = pairs.len() as f32;
    for value in &mut out {
        *value /= n;
    }
    out
}
