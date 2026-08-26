use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    DiagonalGaussianAnomalyDetector, Digest32, Encoder, HashProjectionEncoder, IqCapture, Pipeline,
    PrototypeClassifier, ReferenceIqFeatureExtractor, ReplayStatus, Result, ResultCertificate,
    SdkError, SigintVerificationPolicy, VerificationContext, VerificationDecision,
    VerifiedPipeline,
};

const MAX_GOLDEN_FILE_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_EMBEDDING_DIM: usize = 24;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenSigintSample {
    label: String,
    capture: IqCapture,
}

impl GoldenSigintSample {
    pub fn new(label: impl Into<String>, capture: IqCapture) -> Result<Self> {
        let label = label.into();
        if label.trim().is_empty() || label.len() > 4_096 {
            return Err(SdkError::InvalidArgument(
                "golden SIGINT label must be non-empty and bounded".into(),
            ));
        }
        capture.validate()?;
        Ok(Self { label, capture })
    }

    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn capture(&self) -> &IqCapture {
        &self.capture
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenSigintTrainingFile {
    pub samples: Vec<GoldenSigintSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenSigintScenarioFile {
    pub capture: IqCapture,
    pub expected_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenSigintReport {
    capture_id: String,
    expected_label: Option<String>,
    predicted_label: String,
    probability: f32,
    unknown: bool,
    anomaly_score: Option<f32>,
    uncertainty: f32,
    verification_decision: VerificationDecision,
    ontology_valid: bool,
    certificate: ResultCertificate,
}

impl GoldenSigintReport {
    pub fn capture_id(&self) -> &str {
        &self.capture_id
    }
    pub fn expected_label(&self) -> Option<&str> {
        self.expected_label.as_deref()
    }
    pub fn predicted_label(&self) -> &str {
        &self.predicted_label
    }
    pub fn probability(&self) -> f32 {
        self.probability
    }
    pub fn unknown(&self) -> bool {
        self.unknown
    }
    pub fn anomaly_score(&self) -> Option<f32> {
        self.anomaly_score
    }
    pub fn uncertainty(&self) -> f32 {
        self.uncertainty
    }
    pub fn verification_decision(&self) -> VerificationDecision {
        self.verification_decision
    }
    pub fn ontology_valid(&self) -> bool {
        self.ontology_valid
    }
    pub fn certificate(&self) -> &ResultCertificate {
        &self.certificate
    }
    pub fn correct(&self) -> Option<bool> {
        self.expected_label
            .as_ref()
            .map(|expected| expected == &self.predicted_label)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenSigintEvaluation {
    pub samples: usize,
    pub correct: usize,
    pub accepted: usize,
    pub exact_replays: usize,
    pub accuracy: f32,
    pub accepted_fraction: f32,
    pub exact_replay_fraction: f32,
    pub fixture_metrics_only: bool,
}

#[derive(Clone)]
pub struct GoldenSigintModel {
    extractor: ReferenceIqFeatureExtractor,
    pipeline: VerifiedPipeline,
}

impl GoldenSigintModel {
    pub fn fit(samples: &[GoldenSigintSample], seed: u64) -> Result<Self> {
        if samples.len() < 4 {
            return Err(SdkError::InvalidArgument(
                "golden SIGINT model requires at least four labeled captures".into(),
            ));
        }
        let labels: BTreeSet<&str> = samples.iter().map(|sample| sample.label.as_str()).collect();
        if labels.len() < 2 {
            return Err(SdkError::InvalidArgument(
                "golden SIGINT model requires at least two classes".into(),
            ));
        }
        let extractor = ReferenceIqFeatureExtractor::default();
        let encoder =
            HashProjectionEncoder::new(extractor.feature_dim(), DEFAULT_EMBEDDING_DIM, seed)?;
        let mut labeled_embeddings = Vec::with_capacity(samples.len());
        let mut embeddings = Vec::with_capacity(samples.len());
        for sample in samples {
            let observation = extractor.extract(&sample.capture)?;
            let embedding = encoder.encode(&observation)?;
            embeddings.push(embedding.clone());
            labeled_embeddings.push((embedding, sample.label.clone()));
        }
        let classifier = PrototypeClassifier::fit(&labeled_embeddings)?;
        let anomaly = DiagonalGaussianAnomalyDetector::fit(&embeddings)?;
        let pipeline = Pipeline::new(Arc::new(encoder), Arc::new(classifier), 0.35)?
            .with_anomaly_detector(Arc::new(anomaly))?;
        let training_bytes = serde_json::to_vec(samples)?;
        let context = VerificationContext::new(
            Digest32::from_bytes(&training_bytes),
            Digest32::from_bytes(b"golden-sigint-reference-config-v1"),
            "sigint-reference-ontology-v1",
            "sigint-golden-path-v1",
            seed,
        )?;
        let policy = SigintVerificationPolicy::new(0.35, 1.0, None, 10_000, true)?;
        Ok(Self {
            extractor,
            pipeline: VerifiedPipeline::new(pipeline, context, policy),
        })
    }

    pub fn extractor(&self) -> ReferenceIqFeatureExtractor {
        self.extractor
    }

    pub fn infer(
        &self,
        capture: &IqCapture,
        expected_label: Option<&str>,
    ) -> Result<GoldenSigintReport> {
        let observation = self.extractor.extract(capture)?;
        let verified = self.pipeline.predict(&observation)?;
        let prediction = verified.prediction();
        let top = prediction.top().ok_or_else(|| {
            SdkError::InvalidArgument("golden SIGINT prediction has no class score".into())
        })?;
        Ok(GoldenSigintReport {
            capture_id: capture.id().to_string(),
            expected_label: expected_label.map(ToOwned::to_owned),
            predicted_label: top.label().to_string(),
            probability: top.probability(),
            unknown: prediction.is_unknown(),
            anomaly_score: prediction.anomaly_score(),
            uncertainty: prediction.uncertainty(),
            verification_decision: verified.certificate().decision(),
            ontology_valid: verified.consistency().is_valid(),
            certificate: verified.certificate().clone(),
        })
    }

    pub fn replay(
        &self,
        capture: &IqCapture,
        certificate: &ResultCertificate,
    ) -> Result<(GoldenSigintReport, ReplayStatus)> {
        let observation = self.extractor.extract(capture)?;
        let (verified, status) = self.pipeline.replay(&observation, certificate)?;
        let prediction = verified.prediction();
        let top = prediction.top().ok_or_else(|| {
            SdkError::InvalidArgument("golden SIGINT prediction has no class score".into())
        })?;
        let report = GoldenSigintReport {
            capture_id: capture.id().to_string(),
            expected_label: None,
            predicted_label: top.label().to_string(),
            probability: top.probability(),
            unknown: prediction.is_unknown(),
            anomaly_score: prediction.anomaly_score(),
            uncertainty: prediction.uncertainty(),
            verification_decision: verified.certificate().decision(),
            ontology_valid: verified.consistency().is_valid(),
            certificate: verified.certificate().clone(),
        };
        Ok((report, status))
    }

    pub fn evaluate(&self, samples: &[GoldenSigintSample]) -> Result<GoldenSigintEvaluation> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let mut correct = 0_usize;
        let mut accepted = 0_usize;
        let mut exact_replays = 0_usize;
        for sample in samples {
            let report = self.infer(&sample.capture, Some(&sample.label))?;
            if report.correct() == Some(true) {
                correct = correct.saturating_add(1);
            }
            if report.verification_decision == VerificationDecision::Accept {
                accepted = accepted.saturating_add(1);
            }
            let (_, replay) = self.replay(&sample.capture, report.certificate())?;
            if replay == ReplayStatus::Exact {
                exact_replays = exact_replays.saturating_add(1);
            }
        }
        let count = samples.len() as f32;
        Ok(GoldenSigintEvaluation {
            samples: samples.len(),
            correct,
            accepted,
            exact_replays,
            accuracy: correct as f32 / count,
            accepted_fraction: accepted as f32 / count,
            exact_replay_fraction: exact_replays as f32 / count,
            fixture_metrics_only: true,
        })
    }
}

pub fn load_golden_sigint_training(path: impl AsRef<Path>) -> Result<GoldenSigintTrainingFile> {
    let bytes = read_bounded(path.as_ref(), MAX_GOLDEN_FILE_BYTES)?;
    let file: GoldenSigintTrainingFile = serde_json::from_slice(&bytes)?;
    if file.samples.len() > 65_536 {
        return Err(SdkError::DimensionLimit {
            actual: file.samples.len(),
            max: 65_536,
        });
    }
    for sample in &file.samples {
        GoldenSigintSample::new(sample.label.clone(), sample.capture.clone())?;
    }
    Ok(file)
}

pub fn load_golden_sigint_scenario(path: impl AsRef<Path>) -> Result<GoldenSigintScenarioFile> {
    let bytes = read_bounded(path.as_ref(), MAX_GOLDEN_FILE_BYTES)?;
    let file: GoldenSigintScenarioFile = serde_json::from_slice(&bytes)?;
    file.capture.validate()?;
    if file
        .expected_label
        .as_ref()
        .is_some_and(|label| label.trim().is_empty() || label.len() > 4_096)
    {
        return Err(SdkError::InvalidArgument(
            "expected_label must be non-empty when provided".into(),
        ));
    }
    Ok(file)
}

fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let limit = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut reader = file.take(limit);
    let mut bytes = Vec::with_capacity(max_bytes.min(1024 * 1024));
    reader.read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(SdkError::ArtifactTooLarge {
            actual: bytes.len(),
            max: max_bytes,
        });
    }
    Ok(bytes)
}
