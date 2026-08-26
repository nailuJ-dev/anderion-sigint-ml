use serde::{Deserialize, Serialize};

use crate::classification::squared_distance;
use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedChangePointSegmenter {
    threshold: f32,
    dimension: usize,
}

impl LearnedChangePointSegmenter {
    pub fn fit(training: &[(Embedding, Embedding, bool)]) -> Result<Self> {
        if training.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let dimension = training[0].0.dim();
        let mut points = Vec::with_capacity(training.len());
        for (a, b, boundary) in training {
            if a.dim() != dimension || b.dim() != dimension {
                return Err(SdkError::DimensionMismatch {
                    expected: dimension,
                    actual: a.dim().max(b.dim()),
                });
            }
            points.push((squared_distance(a.values(), b.values())?.sqrt(), *boundary));
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut candidates = Vec::with_capacity(points.len() + 1);
        candidates.push(0.0);
        for pair in points.windows(2) {
            candidates.push((pair[0].0 + pair[1].0) / 2.0);
        }
        candidates.push(points.last().map_or(1.0, |item| item.0 + f32::EPSILON));
        let mut best = (0_usize, candidates[0]);
        for candidate in candidates {
            let correct = points
                .iter()
                .filter(|(distance, boundary)| (*distance >= candidate) == *boundary)
                .count();
            if correct > best.0 {
                best = (correct, candidate);
            }
        }
        Ok(Self {
            threshold: best.1,
            dimension,
        })
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn boundaries(&self, sequence: &[Embedding]) -> Result<Vec<usize>> {
        if sequence.len() < 2 {
            return Ok(Vec::new());
        }
        if sequence.iter().any(|item| item.dim() != self.dimension) {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: sequence[0].dim(),
            });
        }
        let mut boundaries = Vec::new();
        for index in 1..sequence.len() {
            let distance =
                squared_distance(sequence[index - 1].values(), sequence[index].values())?.sqrt();
            if distance >= self.threshold {
                boundaries.push(index);
            }
        }
        Ok(boundaries)
    }
}
