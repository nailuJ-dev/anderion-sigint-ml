use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{Classifier, Encoder, Observation, Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureExplanation {
    pub target_label: String,
    pub base_probability: f32,
    pub importance: Vec<f32>,
}

#[derive(Clone)]
pub struct OcclusionExplainer {
    encoder: Arc<dyn Encoder>,
    classifier: Arc<dyn Classifier>,
    max_features: usize,
}

impl OcclusionExplainer {
    pub fn new(
        encoder: Arc<dyn Encoder>,
        classifier: Arc<dyn Classifier>,
        max_features: usize,
    ) -> Result<Self> {
        if max_features == 0 {
            return Err(SdkError::InvalidArgument(
                "max_features must be positive".into(),
            ));
        }
        if encoder.output_dim() != classifier.input_dim() {
            return Err(SdkError::DimensionMismatch {
                expected: encoder.output_dim(),
                actual: classifier.input_dim(),
            });
        }
        Ok(Self {
            encoder,
            classifier,
            max_features,
        })
    }

    pub fn explain(
        &self,
        observation: &Observation,
        target_label: &str,
    ) -> Result<FeatureExplanation> {
        if observation.features().len() > self.max_features {
            return Err(SdkError::DimensionLimit {
                actual: observation.features().len(),
                max: self.max_features,
            });
        }
        let base_scores = self
            .classifier
            .classify(&self.encoder.encode(observation)?)?;
        let base_probability = probability_for(&base_scores, target_label);
        let mut importance = Vec::with_capacity(observation.features().len());
        for index in 0..observation.features().len() {
            let mut features = observation.features().to_vec();
            features[index] = 0.0;
            let perturbed =
                Observation::new(observation.id(), observation.timestamp_ms(), features)?;
            let scores = self
                .classifier
                .classify(&self.encoder.encode(&perturbed)?)?;
            importance.push(base_probability - probability_for(&scores, target_label));
        }
        Ok(FeatureExplanation {
            target_label: target_label.to_string(),
            base_probability,
            importance,
        })
    }
}

fn probability_for(scores: &[crate::ClassScore], target: &str) -> f32 {
    scores
        .iter()
        .find(|score| score.label() == target)
        .map(crate::ClassScore::probability)
        .unwrap_or(0.0)
}
