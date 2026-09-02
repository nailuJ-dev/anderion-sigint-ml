use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

pub const DEFAULT_MAX_FEATURES: usize = 65_536;
pub const DEFAULT_MAX_EMBEDDING_DIM: usize = 8_192;
/// Maximum byte length accepted for any identifier or class label.
pub const MAX_LABEL_BYTES: usize = 4_096;
/// Maximum number of class scores a single prediction may carry.
pub const MAX_CLASS_SCORES: usize = 4_096;

/// A caller-supplied feature vector.
///
/// Every constructor and every `Deserialize` path enforces the same invariants:
/// bounded identifier, non-empty feature vector, bounded dimension, all values
/// finite. Deserialization always applies [`DEFAULT_MAX_FEATURES`]; use
/// [`Observation::with_limit`] for a different in-process bound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ObservationWire")]
pub struct Observation {
    id: String,
    timestamp_ms: u64,
    features: Vec<f32>,
}

impl Observation {
    pub fn new(id: impl Into<String>, timestamp_ms: u64, features: Vec<f32>) -> Result<Self> {
        Self::with_limit(id, timestamp_ms, features, DEFAULT_MAX_FEATURES)
    }

    pub fn with_limit(
        id: impl Into<String>,
        timestamp_ms: u64,
        features: Vec<f32>,
        max_features: usize,
    ) -> Result<Self> {
        let id = id.into();
        validate_label("observation id", &id)?;
        validate_vector(&features, max_features)?;
        Ok(Self {
            id,
            timestamp_ms,
            features,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
    pub fn features(&self) -> &[f32] {
        &self.features
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationWire {
    id: String,
    timestamp_ms: u64,
    features: Vec<f32>,
}

impl TryFrom<ObservationWire> for Observation {
    type Error = SdkError;

    fn try_from(wire: ObservationWire) -> Result<Self> {
        Self::new(wire.id, wire.timestamp_ms, wire.features)
    }
}

/// A bounded, finite embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "EmbeddingWire")]
pub struct Embedding {
    values: Vec<f32>,
}

impl Embedding {
    pub fn new(values: Vec<f32>) -> Result<Self> {
        Self::with_limit(values, DEFAULT_MAX_EMBEDDING_DIM)
    }

    pub fn with_limit(values: Vec<f32>, max_dim: usize) -> Result<Self> {
        validate_vector(&values, max_dim)?;
        Ok(Self { values })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingWire {
    values: Vec<f32>,
}

impl TryFrom<EmbeddingWire> for Embedding {
    type Error = SdkError;

    fn try_from(wire: EmbeddingWire) -> Result<Self> {
        Self::new(wire.values)
    }
}

/// A class label with its probability in `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ClassScoreWire")]
pub struct ClassScore {
    label: String,
    probability: f32,
}

impl ClassScore {
    pub fn new(label: impl Into<String>, probability: f32) -> Result<Self> {
        validate_probability(probability)?;
        let label = label.into();
        validate_label("class label", &label)?;
        Ok(Self { label, probability })
    }

    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn probability(&self) -> f32 {
        self.probability
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassScoreWire {
    label: String,
    probability: f32,
}

impl TryFrom<ClassScoreWire> for ClassScore {
    type Error = SdkError;

    fn try_from(wire: ClassScoreWire) -> Result<Self> {
        Self::new(wire.label, wire.probability)
    }
}

/// A pipeline result.
///
/// `scores` is always sorted by descending probability and always carries
/// unique labels, on construction **and** on deserialization.
/// [`Prediction::top`] therefore always returns the highest-scoring class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PredictionWire")]
pub struct Prediction {
    scores: Vec<ClassScore>,
    unknown: bool,
    anomaly_score: Option<f32>,
    embedding: Embedding,
    uncertainty: f32,
}

impl Prediction {
    pub fn new(
        mut scores: Vec<ClassScore>,
        unknown: bool,
        anomaly_score: Option<f32>,
        embedding: Embedding,
        uncertainty: f32,
    ) -> Result<Self> {
        if scores.is_empty() {
            return Err(SdkError::InvalidArgument(
                "prediction requires at least one class score".into(),
            ));
        }
        if scores.len() > MAX_CLASS_SCORES {
            return Err(SdkError::DimensionLimit {
                actual: scores.len(),
                max: MAX_CLASS_SCORES,
            });
        }
        {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for score in &scores {
                if !seen.insert(score.label()) {
                    return Err(SdkError::InvalidArgument(
                        "prediction contains duplicate class labels".into(),
                    ));
                }
            }
        }
        validate_probability(uncertainty)?;
        if let Some(value) = anomaly_score {
            validate_probability(value)?;
        }
        // Deterministic total order: probability descending, then label ascending.
        // The label tiebreak makes `top()` independent of producer iteration order.
        scores.sort_by(|a, b| {
            b.probability
                .total_cmp(&a.probability)
                .then_with(|| a.label.cmp(&b.label))
        });
        Ok(Self {
            scores,
            unknown,
            anomaly_score,
            embedding,
            uncertainty,
        })
    }

    pub fn scores(&self) -> &[ClassScore] {
        &self.scores
    }
    pub fn top(&self) -> Option<&ClassScore> {
        self.scores.first()
    }
    pub fn is_unknown(&self) -> bool {
        self.unknown
    }
    pub fn anomaly_score(&self) -> Option<f32> {
        self.anomaly_score
    }
    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }
    pub fn uncertainty(&self) -> f32 {
        self.uncertainty
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictionWire {
    scores: Vec<ClassScore>,
    unknown: bool,
    anomaly_score: Option<f32>,
    embedding: Embedding,
    uncertainty: f32,
}

impl TryFrom<PredictionWire> for Prediction {
    type Error = SdkError;

    fn try_from(wire: PredictionWire) -> Result<Self> {
        Self::new(
            wire.scores,
            wire.unknown,
            wire.anomaly_score,
            wire.embedding,
            wire.uncertainty,
        )
    }
}

pub(crate) fn validate_vector(values: &[f32], max_dim: usize) -> Result<()> {
    if values.is_empty() {
        return Err(SdkError::EmptyFeatures);
    }
    if values.len() > max_dim {
        return Err(SdkError::DimensionLimit {
            actual: values.len(),
            max: max_dim,
        });
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SdkError::NonFiniteValue { index });
    }
    Ok(())
}

pub(crate) fn validate_probability(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(SdkError::InvalidProbability(value));
    }
    Ok(())
}

pub(crate) fn validate_label(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(SdkError::InvalidArgument(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > MAX_LABEL_BYTES {
        return Err(SdkError::DimensionLimit {
            actual: value.len(),
            max: MAX_LABEL_BYTES,
        });
    }
    Ok(())
}
