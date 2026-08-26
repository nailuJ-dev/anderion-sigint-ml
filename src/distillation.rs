use std::collections::BTreeMap;

use crate::classification::softmax_scores;
use crate::{ClassScore, Classifier, Embedding, Result, SdkError};

#[derive(Debug, Clone)]
pub struct SoftLabelSample {
    embedding: Embedding,
    scores: Vec<ClassScore>,
}

impl SoftLabelSample {
    pub fn new(embedding: Embedding, scores: Vec<ClassScore>) -> Result<Self> {
        if scores.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let sum: f32 = scores.iter().map(ClassScore::probability).sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "soft-label probabilities must have positive finite mass".into(),
            ));
        }
        Ok(Self { embedding, scores })
    }
}

#[derive(Debug, Clone)]
pub struct DistilledPrototypeClassifier {
    dimension: usize,
    prototypes: BTreeMap<String, Vec<f32>>,
    temperature: f32,
}

impl DistilledPrototypeClassifier {
    pub fn fit(samples: &[SoftLabelSample], temperature: f32) -> Result<Self> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "temperature must be finite and positive".into(),
            ));
        }
        let dimension = samples[0].embedding.dim();
        let mut sums: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut masses: BTreeMap<String, f32> = BTreeMap::new();
        for sample in samples {
            if sample.embedding.dim() != dimension {
                return Err(SdkError::DimensionMismatch {
                    expected: dimension,
                    actual: sample.embedding.dim(),
                });
            }
            for score in &sample.scores {
                let sum = sums
                    .entry(score.label().to_string())
                    .or_insert_with(|| vec![0.0; dimension]);
                for (dst, value) in sum.iter_mut().zip(sample.embedding.values()) {
                    *dst += score.probability() * *value;
                }
                *masses.entry(score.label().to_string()).or_insert(0.0) += score.probability();
            }
        }
        let mut prototypes = BTreeMap::new();
        for (label, mut sum) in sums {
            let mass = masses.get(&label).copied().unwrap_or(0.0);
            if mass <= f32::EPSILON {
                continue;
            }
            for value in &mut sum {
                *value /= mass;
            }
            prototypes.insert(label, sum);
        }
        if prototypes.is_empty()
            || prototypes
                .values()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(SdkError::InvalidArgument(
                "distillation produced invalid class prototypes".into(),
            ));
        }
        Ok(Self {
            dimension,
            prototypes,
            temperature,
        })
    }
}

impl Classifier for DistilledPrototypeClassifier {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        let logits = self
            .prototypes
            .iter()
            .map(|(label, prototype)| {
                let distance: f32 = embedding
                    .values()
                    .iter()
                    .zip(prototype)
                    .map(|(x, y)| {
                        let d = x - y;
                        d * d
                    })
                    .sum();
                (label.clone(), -distance / self.temperature)
            })
            .collect::<Vec<_>>();
        softmax_scores(&logits)
    }

    fn input_dim(&self) -> usize {
        self.dimension
    }
}
