use std::collections::BTreeMap;

use crate::classification::softmax_scores;
use crate::{ClassScore, Classifier, Embedding, Result, SdkError};

#[derive(Debug, Clone)]
pub struct EmbeddingZeroShotClassifier {
    prototypes: BTreeMap<String, Embedding>,
    dimension: usize,
    temperature: f32,
}

impl EmbeddingZeroShotClassifier {
    pub fn new(prototypes: Vec<(String, Embedding)>, temperature: f32) -> Result<Self> {
        if prototypes.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "temperature must be finite and positive".into(),
            ));
        }
        let dimension = prototypes[0].1.dim();
        let mut map = BTreeMap::new();
        for (label, embedding) in prototypes {
            if label.trim().is_empty() {
                return Err(SdkError::InvalidArgument(
                    "prototype label must be non-empty".into(),
                ));
            }
            if embedding.dim() != dimension {
                return Err(SdkError::DimensionMismatch {
                    expected: dimension,
                    actual: embedding.dim(),
                });
            }
            if map.insert(label, embedding).is_some() {
                return Err(SdkError::InvalidArgument(
                    "prototype labels must be unique".into(),
                ));
            }
        }
        Ok(Self {
            prototypes: map,
            dimension,
            temperature,
        })
    }
}

impl Classifier for EmbeddingZeroShotClassifier {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        let mut logits = Vec::with_capacity(self.prototypes.len());
        for (label, prototype) in &self.prototypes {
            logits.push((
                label.clone(),
                cosine(embedding.values(), prototype.values())? / self.temperature,
            ));
        }
        softmax_scores(&logits)
    }

    fn input_dim(&self) -> usize {
        self.dimension
    }
}

fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(SdkError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        return Ok(0.0);
    }
    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}
