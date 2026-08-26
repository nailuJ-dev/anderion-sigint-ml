use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftReport {
    pub score: f32,
    pub drifted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMonitor {
    baseline_mean: Vec<f32>,
    baseline_std: Vec<f32>,
    threshold: f32,
    epsilon: f32,
}

impl DriftMonitor {
    pub fn fit(baseline: &[Embedding], threshold: f32) -> Result<Self> {
        if baseline.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "drift threshold must be finite and positive".into(),
            ));
        }
        let dim = baseline[0].dim();
        if baseline.iter().any(|sample| sample.dim() != dim) {
            return Err(SdkError::InvalidArgument(
                "baseline samples must share a dimension".into(),
            ));
        }
        let mut mean = vec![0.0_f32; dim];
        for sample in baseline {
            for (dst, src) in mean.iter_mut().zip(sample.values()) {
                *dst += *src;
            }
        }
        let n = baseline.len() as f32;
        for value in &mut mean {
            *value /= n;
        }
        let mut std = vec![0.0_f32; dim];
        for sample in baseline {
            for ((dst, src), avg) in std.iter_mut().zip(sample.values()).zip(&mean) {
                let delta = *src - *avg;
                *dst += delta * delta;
            }
        }
        for value in &mut std {
            *value = (*value / n).sqrt();
        }
        Ok(Self {
            baseline_mean: mean,
            baseline_std: std,
            threshold,
            epsilon: 1e-3,
        })
    }

    pub fn evaluate(&self, current: &[Embedding]) -> Result<DriftReport> {
        if current.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if current
            .iter()
            .any(|sample| sample.dim() != self.baseline_mean.len())
        {
            return Err(SdkError::DimensionMismatch {
                expected: self.baseline_mean.len(),
                actual: current[0].dim(),
            });
        }
        let mut mean = vec![0.0_f32; self.baseline_mean.len()];
        for sample in current {
            for (dst, src) in mean.iter_mut().zip(sample.values()) {
                *dst += *src;
            }
        }
        for value in &mut mean {
            *value /= current.len() as f32;
        }
        let score = mean
            .iter()
            .zip(&self.baseline_mean)
            .zip(&self.baseline_std)
            .map(|((cur, base), std)| (cur - base).abs() / (*std + self.epsilon))
            .sum::<f32>()
            / mean.len() as f32;
        Ok(DriftReport {
            score,
            drifted: score >= self.threshold,
        })
    }
}
