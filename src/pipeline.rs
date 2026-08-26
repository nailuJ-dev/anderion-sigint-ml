use std::sync::Arc;

use crate::{
    AnomalyDetector, Calibrator, Classifier, Encoder, Observation, Prediction, Result, SdkError,
    normalized_entropy,
};

#[derive(Clone)]
pub struct Pipeline {
    encoder: Arc<dyn Encoder>,
    classifier: Arc<dyn Classifier>,
    anomaly_detector: Option<Arc<dyn AnomalyDetector>>,
    calibrator: Option<Arc<dyn Calibrator>>,
    unknown_threshold: f32,
}

impl Pipeline {
    pub fn new(
        encoder: Arc<dyn Encoder>,
        classifier: Arc<dyn Classifier>,
        unknown_threshold: f32,
    ) -> Result<Self> {
        if encoder.output_dim() != classifier.input_dim() {
            return Err(SdkError::DimensionMismatch {
                expected: encoder.output_dim(),
                actual: classifier.input_dim(),
            });
        }
        if !unknown_threshold.is_finite() || !(0.0..=1.0).contains(&unknown_threshold) {
            return Err(SdkError::InvalidProbability(unknown_threshold));
        }
        Ok(Self {
            encoder,
            classifier,
            anomaly_detector: None,
            calibrator: None,
            unknown_threshold,
        })
    }

    pub fn with_anomaly_detector(mut self, detector: Arc<dyn AnomalyDetector>) -> Result<Self> {
        if detector.input_dim() != self.encoder.output_dim() {
            return Err(SdkError::DimensionMismatch {
                expected: self.encoder.output_dim(),
                actual: detector.input_dim(),
            });
        }
        self.anomaly_detector = Some(detector);
        Ok(self)
    }

    pub fn with_calibrator(mut self, calibrator: Arc<dyn Calibrator>) -> Self {
        self.calibrator = Some(calibrator);
        self
    }

    pub fn predict_batch(&self, observations: &[Observation]) -> Result<Vec<Prediction>> {
        if observations.len() > 65_536 {
            return Err(SdkError::DimensionLimit {
                actual: observations.len(),
                max: 65_536,
            });
        }
        observations
            .iter()
            .map(|observation| self.predict(observation))
            .collect()
    }

    pub fn predict_stream<'a, I>(
        &'a self,
        observations: I,
    ) -> impl Iterator<Item = Result<Prediction>> + 'a
    where
        I: IntoIterator<Item = Observation> + 'a,
    {
        observations
            .into_iter()
            .map(|observation| self.predict(&observation))
    }

    pub fn predict(&self, observation: &Observation) -> Result<Prediction> {
        if observation.features().len() != self.encoder.input_dim() {
            return Err(SdkError::DimensionMismatch {
                expected: self.encoder.input_dim(),
                actual: observation.features().len(),
            });
        }
        let embedding = self.encoder.encode(observation)?;
        let mut scores = self.classifier.classify(&embedding)?;
        if let Some(calibrator) = &self.calibrator {
            scores = calibrator.calibrate(&scores)?;
        }
        let uncertainty = normalized_entropy(&scores)?;
        let unknown = scores
            .first()
            .map(|score| score.probability() < self.unknown_threshold)
            .unwrap_or(true);
        let anomaly_score = self
            .anomaly_detector
            .as_ref()
            .map(|detector| detector.anomaly_score(&embedding))
            .transpose()?;
        Prediction::new(scores, unknown, anomaly_score, embedding, uncertainty)
    }
}
