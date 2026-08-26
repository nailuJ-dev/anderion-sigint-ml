use std::collections::BTreeMap;

use crate::temporal::normalize;
use crate::{ClassScore, Result, SdkError};

#[derive(Debug, Clone)]
pub struct WeightedEnsemble {
    weights: Vec<f32>,
}

impl WeightedEnsemble {
    pub fn new(weights: Vec<f32>) -> Result<Self> {
        if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(SdkError::InvalidArgument(
                "ensemble weights must be finite and non-negative".into(),
            ));
        }
        if weights.iter().sum::<f32>() <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "ensemble requires positive total weight".into(),
            ));
        }
        Ok(Self { weights })
    }

    pub fn combine(&self, predictions: &[Vec<ClassScore>]) -> Result<Vec<ClassScore>> {
        if predictions.len() != self.weights.len() {
            return Err(SdkError::DimensionMismatch {
                expected: self.weights.len(),
                actual: predictions.len(),
            });
        }
        let total_weight: f32 = self.weights.iter().sum();
        let mut totals: BTreeMap<String, f32> = BTreeMap::new();
        for (prediction, weight) in predictions.iter().zip(&self.weights) {
            for score in prediction {
                *totals.entry(score.label().to_string()).or_insert(0.0) +=
                    *weight * score.probability();
            }
        }
        normalize(
            totals
                .into_iter()
                .map(|(label, value)| (label, value / total_weight))
                .collect(),
        )
    }
}
