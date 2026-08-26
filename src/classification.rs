use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{ClassScore, Classifier, Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeClassifier {
    dimension: usize,
    prototypes: BTreeMap<String, Vec<f32>>,
    counts: BTreeMap<String, u64>,
    temperature: f32,
}

impl PrototypeClassifier {
    pub fn fit(samples: &[(Embedding, String)]) -> Result<Self> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let dimension = samples[0].0.dim();
        let mut sums: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for (embedding, label) in samples {
            if embedding.dim() != dimension {
                return Err(SdkError::DimensionMismatch {
                    expected: dimension,
                    actual: embedding.dim(),
                });
            }
            if label.trim().is_empty() {
                return Err(SdkError::InvalidArgument(
                    "class label must not be empty".into(),
                ));
            }
            let sum = sums
                .entry(label.clone())
                .or_insert_with(|| vec![0.0; dimension]);
            for (dst, src) in sum.iter_mut().zip(embedding.values()) {
                *dst += *src;
            }
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        let mut prototypes = BTreeMap::new();
        for (label, mut sum) in sums {
            let count = *counts
                .get(&label)
                .ok_or_else(|| SdkError::InvalidArgument("missing class count".into()))?
                as f32;
            for value in &mut sum {
                *value /= count;
            }
            prototypes.insert(label, sum);
        }
        Ok(Self {
            dimension,
            prototypes,
            counts,
            temperature: 1.0,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 || self.dimension > 8_192 || self.prototypes.is_empty() {
            return Err(SdkError::InvalidArgument(
                "invalid prototype classifier dimensions or empty classes".into(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "classifier temperature must be finite and positive".into(),
            ));
        }
        for (label, prototype) in &self.prototypes {
            if label.trim().is_empty()
                || prototype.len() != self.dimension
                || prototype.iter().any(|v| !v.is_finite())
            {
                return Err(SdkError::InvalidArgument(
                    "invalid prototype classifier payload".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "temperature must be finite and positive".into(),
            ));
        }
        self.temperature = temperature;
        Ok(self)
    }

    pub fn prototypes(&self) -> &BTreeMap<String, Vec<f32>> {
        &self.prototypes
    }
    pub(crate) fn counts(&self) -> &BTreeMap<String, u64> {
        &self.counts
    }
    pub(crate) fn temperature(&self) -> f32 {
        self.temperature
    }
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Classifier for PrototypeClassifier {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        let mut logits = Vec::with_capacity(self.prototypes.len());
        for (label, prototype) in &self.prototypes {
            let distance = squared_distance(embedding.values(), prototype)?;
            logits.push((label.clone(), -distance / self.temperature));
        }
        softmax_scores(&logits)
    }

    fn input_dim(&self) -> usize {
        self.dimension
    }
}

pub(crate) fn squared_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(SdkError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum())
}

pub(crate) fn softmax_scores(logits: &[(String, f32)]) -> Result<Vec<ClassScore>> {
    if logits.is_empty() {
        return Err(SdkError::InvalidArgument(
            "softmax requires at least one class".into(),
        ));
    }
    if logits.iter().any(|(_, value)| !value.is_finite()) {
        return Err(SdkError::InvalidArgument(
            "softmax logits must be finite".into(),
        ));
    }
    let max = logits
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|(_, value)| (*value - max).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SdkError::InvalidArgument(
            "softmax normalization failed".into(),
        ));
    }
    let mut scores = Vec::with_capacity(logits.len());
    for ((label, _), exp) in logits.iter().zip(exps) {
        scores.push(ClassScore::new(label.clone(), exp / sum)?);
    }
    scores.sort_by(|a, b| b.probability().total_cmp(&a.probability()));
    Ok(scores)
}
