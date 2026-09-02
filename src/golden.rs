use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    DiagonalGaussianAnomalyDetector, Digest32, Encoder, HashProjectionEncoder, IqCapture, Pipeline,
    PrototypeClassifier, ReferenceIqFeatureExtractor, ReplayStatus, Result, ResultCertificate,
    SdkError, SigintVerificationPolicy, VerificationContext, VerificationDecision,
    VerifiedPipeline,
};

use crate::io::read_bounded;

const MAX_GOLDEN_FILE_BYTES: usize = 32 * 1024 * 1024;
const MAX_GOLDEN_SAMPLES: usize = 65_536;
const DEFAULT_EMBEDDING_DIM: usize = 24;
const GOLDEN_UNKNOWN_THRESHOLD: f32 = 0.35;
const GOLDEN_CONFIG_VERSION: &str = "golden-sigint-reference-config-v3";
const GOLDEN_ONTOLOGY_VERSION: &str = "sigint-reference-ontology-v1";
const GOLDEN_PIPELINE_VERSION: &str = "sigint-golden-path-v3";

/// Everything that changes the numeric behaviour of the reference model.
///
/// Serialized and hashed into [`VerificationContext::config_digest`] so that two
/// runs with different hyper-parameters can never present matching context
/// digests. Add a field here whenever a new knob is introduced.
#[derive(Debug, Serialize)]
struct GoldenSigintConfig<'a> {
    config_version: &'a str,
    extractor: ReferenceIqFeatureExtractor,
    embedding_dim: usize,
    unknown_threshold: f32,
    policy: &'a SigintVerificationPolicy,
    anomaly_detector: &'a str,
}

/// Default verification policy for the reference Golden Path.
///
/// `max_uncertainty` is a real gate: normalized entropy above this value forces
/// abstention. Tune it against your own fixtures rather than treating it as a
/// constant of nature.
pub fn default_golden_policy() -> Result<SigintVerificationPolicy> {
    SigintVerificationPolicy::new(GOLDEN_UNKNOWN_THRESHOLD, 0.95, None, 10_000, true)
}

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

    /// Validate in place, without cloning the underlying capture.
    pub fn validate(&self) -> Result<()> {
        if self.label.trim().is_empty() || self.label.len() > 4_096 {
            return Err(SdkError::InvalidArgument(
                "golden SIGINT label must be non-empty and bounded".into(),
            ));
        }
        self.capture.validate()
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
        Self::fit_with_policy(samples, seed, default_golden_policy()?)
    }

    pub fn fit_with_policy(
        samples: &[GoldenSigintSample],
        seed: u64,
        policy: SigintVerificationPolicy,
    ) -> Result<Self> {
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
        let pipeline = Pipeline::new(
            Arc::new(encoder),
            Arc::new(classifier),
            GOLDEN_UNKNOWN_THRESHOLD,
        )?
        .with_anomaly_detector(Arc::new(anomaly))?;
        let training_bytes = serde_json::to_vec(samples)?;
        let config = GoldenSigintConfig {
            config_version: GOLDEN_CONFIG_VERSION,
            extractor,
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            unknown_threshold: GOLDEN_UNKNOWN_THRESHOLD,
            policy: &policy,
            anomaly_detector: "diagonal-gaussian-v1",
        };
        let config_bytes = serde_json::to_vec(&config)?;
        let context = VerificationContext::new(
            Digest32::from_bytes(&training_bytes),
            Digest32::from_bytes(&config_bytes),
            GOLDEN_ONTOLOGY_VERSION,
            GOLDEN_PIPELINE_VERSION,
            seed,
        )?;
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
    if file.samples.len() > MAX_GOLDEN_SAMPLES {
        return Err(SdkError::DimensionLimit {
            actual: file.samples.len(),
            max: MAX_GOLDEN_SAMPLES,
        });
    }
    // `GoldenSigintSample` revalidates on deserialization; this is a cheap
    // defence-in-depth pass that borrows instead of deep-cloning every capture.
    for sample in &file.samples {
        sample.validate()?;
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
