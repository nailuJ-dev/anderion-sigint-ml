use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

pub const DEFAULT_MAX_FEATURES: usize = 65_536;
pub const DEFAULT_MAX_EMBEDDING_DIM: usize = 8_192;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        validate_vector(&features, max_features)?;
        Ok(Self {
            id: id.into(),
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassScore {
    label: String,
    probability: f32,
}

impl ClassScore {
    pub fn new(label: impl Into<String>, probability: f32) -> Result<Self> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(SdkError::InvalidProbability(probability));
        }
        let label = label.into();
        if label.trim().is_empty() {
            return Err(SdkError::InvalidArgument("label must not be empty".into()));
        }
        Ok(Self { label, probability })
    }

    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn probability(&self) -> f32 {
        self.probability
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        if !uncertainty.is_finite() || !(0.0..=1.0).contains(&uncertainty) {
            return Err(SdkError::InvalidProbability(uncertainty));
        }
        if let Some(value) = anomaly_score {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(SdkError::InvalidProbability(value));
            }
        }
        scores.sort_by(|a, b| b.probability.total_cmp(&a.probability));
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
