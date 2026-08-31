#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Anderion SIGINT ML is a standalone, sensor-agnostic machine-learning SDK for
//! user-supplied signal observations. It has no dependency on external model services
//! or non-public runtime components and contains no signal-control functionality.

mod active_learning;
mod adaptation;
mod anomaly;
mod artifact;
mod attention;
mod autoencoder;
mod backend;
mod benchmark;
mod calibration;
mod classification;
mod clustering;
mod continual;
mod components;
mod contrastive;
mod dataset;
mod distillation;
mod drift;
mod edge;
mod embedding;
mod ensemble;
mod emitter_discovery;
mod evidence_graph;
mod error;
mod evaluation;
mod explainability;
mod foundation;
mod golden;
mod iq;
mod metadata;
mod metric_learning;
mod model;
mod ontology;
mod open_set;
mod pattern;
mod pipeline;
mod pruning;
mod qat;
mod quantization;
mod receiver;
mod segmentation;
mod self_supervised;
mod sequence;
#[cfg(feature = "server")]
pub mod service;
mod similarity;
mod spectrum_encoder;
mod synthetic_adapter;
mod temporal;
mod types;
mod uncertainty;
mod verification;
mod verified_pipeline;
mod zero_shot;

pub use active_learning::{ActiveLearningCandidate, select_uncertain_diverse};
pub use adaptation::MeanVarianceAdapter;
pub use anomaly::DiagonalGaussianAnomalyDetector;
pub use artifact::{ArtifactManifest, ArtifactPolicy, load_verified_payload, verify_payload};
pub use attention::{TemporalSelfAttentionEncoder, TemporalTransformerEncoder};
pub use autoencoder::VarianceAutoencoder;
pub use calibration::TemperatureScaler;
pub use classification::PrototypeClassifier;
pub use clustering::{KMeansResult, kmeans};
pub use continual::OnlinePrototypeClassifier;
pub use components::{DenseSpectrumExtractor, SignalComponent};
pub use contrastive::ContrastiveProjector;
pub use dataset::{DatasetRow, DatasetSplit, grouped_split};
pub use distillation::{DistilledPrototypeClassifier, SoftLabelSample};
pub use drift::{DriftMonitor, DriftReport};
pub use edge::{
    EdgeEncoderBenchmark, EdgeParameterProfile, benchmark_encoder, profile_parameter_matrix,
};
pub use embedding::HashProjectionEncoder;
pub use ensemble::WeightedEnsemble;
pub use emitter_discovery::{EmitterDiscoverySession, EmitterHypothesis, EmitterHypothesisStatus};
pub use evidence_graph::{EvidenceEdge, EvidenceEdgeKind, EvidenceNode, SessionEvidenceGraph, WhyResult};
pub use error::{Result, SdkError};
pub use evaluation::{ClassificationMetrics, classification_metrics};
pub use foundation::{FoundationModel, FoundationPooler};
pub use metric_learning::DiagonalMetricLearner;
pub use model::{AnomalyDetector, Calibrator, Classifier, Encoder};
pub use open_set::NearestPrototypeOod;
pub use pipeline::Pipeline;
pub use pruning::{magnitude_prune, sparsity};
pub use qat::QuantizationAwarePrototypeClassifier;
pub use quantization::{QuantizedEmbedding, SymmetricQuantizer};
pub use receiver::{ReceiverProfile, cross_receiver_consistency, normalize_receiver_capture};
pub use similarity::{SimilarityHit, SimilarityIndex};
pub use spectrum_encoder::{ReferenceSpectrumEncoder, SpectrumEncoder};
pub use synthetic_adapter::{SyntheticDataAdapter, SyntheticFeatureGenerator};
pub use temporal::TemporalVote;
pub use types::{ClassScore, Embedding, Observation, Prediction};
pub use uncertainty::normalized_entropy;
pub use zero_shot::EmbeddingZeroShotClassifier;

pub use backend::{BackendKind, ComputeBackend, CpuBackend};
pub use benchmark::{BenchmarkConfig, BenchmarkReport, benchmark_pipeline};
pub use dataset::DatasetManifest;
pub use explainability::{FeatureExplanation, OcclusionExplainer};
pub use metadata::ModelCard;
pub use segmentation::LearnedChangePointSegmenter;
pub use self_supervised::MaskedContextPretrainer;
pub use sequence::SequenceClassifier;

pub use ontology::{
    ConceptKind, ConsistencyReport, ConsistencyViolation, OntologyGraph, OntologyNode,
    OntologyRelation, RelationKind, semantic_graph_for_prediction,
};
pub use pattern::{
    CooccurrencePattern, PatternEngine, PatternEvent, PatternToken, RecurringPattern,
    pattern_event_from_prediction,
};
pub use verification::{
    DeterministicVerifier, Digest32, ReplayStatus, ResultCertificate, SigintVerificationPolicy,
    VerificationContext, VerificationDecision,
};
pub use verified_pipeline::{VerifiedPipeline, VerifiedPrediction};

pub use golden::{
    GoldenSigintEvaluation, GoldenSigintModel, GoldenSigintReport, GoldenSigintSample,
    GoldenSigintScenarioFile, GoldenSigintTrainingFile, load_golden_sigint_scenario,
    load_golden_sigint_training,
};
pub use iq::{
    IqCapture, IqSample, MAX_IQ_SAMPLES, MAX_REFERENCE_SPECTRUM_BINS, ReferenceIqFeatureExtractor,
};
