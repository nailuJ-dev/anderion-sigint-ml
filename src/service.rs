use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::{
    ArtifactPolicy, HashProjectionEncoder, Observation, Pipeline, Prediction, PrototypeClassifier,
    Result, SdkError, load_verified_payload,
};

/// Maximum number of inference requests executed concurrently.
pub const DEFAULT_MAX_CONCURRENT_INFERENCES: usize = 32;
/// Wall-clock budget for a single inference request.
pub const DEFAULT_INFERENCE_TIMEOUT: Duration = Duration::from_secs(5);
/// Maximum accepted request body.
pub const DEFAULT_MAX_BODY_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceModelBundle {
    pub encoder: HashProjectionEncoder,
    pub classifier: PrototypeClassifier,
    pub unknown_threshold: f32,
}

impl ReferenceModelBundle {
    pub fn into_pipeline(self) -> Result<Pipeline> {
        self.encoder.validate()?;
        self.classifier.validate()?;
        Pipeline::new(
            Arc::new(self.encoder),
            Arc::new(self.classifier),
            self.unknown_threshold,
        )
    }
}

pub fn load_reference_bundle(
    manifest_path: impl AsRef<Path>,
    payload_path: impl AsRef<Path>,
) -> Result<ReferenceModelBundle> {
    let policy = ArtifactPolicy::default();
    let (_, payload) = load_verified_payload(manifest_path, payload_path, &policy)?;
    let bundle: ReferenceModelBundle = serde_json::from_slice(&payload)?;
    if !bundle.unknown_threshold.is_finite() || !(0.0..=1.0).contains(&bundle.unknown_threshold) {
        return Err(SdkError::InvalidProbability(bundle.unknown_threshold));
    }
    Ok(bundle)
}

#[derive(Clone)]
pub struct ServiceState {
    pipeline: Arc<Pipeline>,
    permits: Arc<Semaphore>,
    timeout: Duration,
}

impl ServiceState {
    pub fn new(pipeline: Pipeline) -> Self {
        Self::with_limits(
            pipeline,
            DEFAULT_MAX_CONCURRENT_INFERENCES,
            DEFAULT_INFERENCE_TIMEOUT,
        )
    }

    /// Build a state with explicit resource limits.
    ///
    /// `max_concurrent_inferences` bounds how many CPU-bound inferences may run
    /// at once; excess requests are rejected with 503 rather than queued.
    /// `timeout` bounds client-visible latency.
    pub fn with_limits(
        pipeline: Pipeline,
        max_concurrent_inferences: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
            permits: Arc::new(Semaphore::new(max_concurrent_inferences.max(1))),
            timeout,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictRequest {
    pub id: String,
    pub timestamp_ms: u64,
    pub features: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub fn build_router(state: ServiceState) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/predict", post(predict))
        .layer(DefaultBodyLimit::max(DEFAULT_MAX_BODY_BYTES))
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn predict(
    State(state): State<ServiceState>,
    Json(request): Json<PredictRequest>,
) -> std::result::Result<Json<Prediction>, (StatusCode, Json<ErrorResponse>)> {
    let observation = Observation::new(request.id, request.timestamp_ms, request.features)
        .map_err(bad_request)?;

    // Reject rather than queue: an unbounded queue turns a load spike into
    // unbounded memory growth and unbounded latency.
    let permit = state
        .permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| overloaded())?;

    let pipeline = Arc::clone(&state.pipeline);
    // Inference is CPU-bound; running it inline would block an async worker.
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        pipeline.predict(&observation)
    });

    match tokio::time::timeout(state.timeout, task).await {
        Err(_elapsed) => Err(timed_out()),
        Ok(Err(_join_error)) => Err(inference_failed()),
        Ok(Ok(Err(error))) => Err(inference_rejected(error)),
        Ok(Ok(Ok(prediction))) => Ok(Json(prediction)),
    }
}

fn bad_request(error: SdkError) -> (StatusCode, Json<ErrorResponse>) {
    // Validation errors describe the caller-supplied payload only.
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn inference_rejected(error: SdkError) -> (StatusCode, Json<ErrorResponse>) {
    // Shape mismatches between the request and the loaded model are the
    // caller's problem and safe to describe; anything else is not.
    let message = match &error {
        SdkError::DimensionMismatch { .. } | SdkError::DimensionLimit { .. } => error.to_string(),
        _ => "inference rejected the request".to_string(),
    };
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(ErrorResponse { error: message }),
    )
}

fn inference_failed() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: "internal error".to_string(),
        }),
    )
}

fn timed_out() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::GATEWAY_TIMEOUT,
        Json(ErrorResponse {
            error: "inference timed out".to_string(),
        }),
    )
}

fn overloaded() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "too many concurrent inferences".to_string(),
        }),
    )
}
