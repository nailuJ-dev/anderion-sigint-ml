use serde::{Deserialize, Serialize};

use crate::classification::squared_distance;
use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestPrototypeOod {
    prototypes: Vec<Embedding>,
    max_distance: f32,
}

impl NearestPrototypeOod {
    pub fn new(prototypes: Vec<Embedding>, max_distance: f32) -> Result<Self> {
        if prototypes.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if !max_distance.is_finite() || max_distance < 0.0 {
            return Err(SdkError::InvalidArgument(
                "max_distance must be finite and non-negative".into(),
            ));
        }
        let dim = prototypes[0].dim();
        if prototypes.iter().any(|p| p.dim() != dim) {
            return Err(SdkError::InvalidArgument(
                "all OOD prototypes must share a dimension".into(),
            ));
        }
        Ok(Self {
            prototypes,
            max_distance,
        })
    }

    pub fn nearest_distance(&self, embedding: &Embedding) -> Result<f32> {
        if embedding.dim() != self.prototypes[0].dim() {
            return Err(SdkError::DimensionMismatch {
                expected: self.prototypes[0].dim(),
                actual: embedding.dim(),
            });
        }
        self.prototypes
            .iter()
            .map(|prototype| {
                squared_distance(embedding.values(), prototype.values()).map(f32::sqrt)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min_by(f32::total_cmp)
            .ok_or(SdkError::EmptyDataset)
    }

    pub fn is_unknown(&self, embedding: &Embedding) -> Result<bool> {
        Ok(self.nearest_distance(embedding)? > self.max_distance)
    }
}
