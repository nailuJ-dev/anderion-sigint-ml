use std::path::Path;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::{
    ArtifactPolicy, HashProjectionEncoder, Observation, Pipeline, Prediction, PrototypeClassifier,
    Result, SdkError, load_verified_payload,
};

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
}

impl ServiceState {
    pub fn new(pipeline: Pipeline) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
        }
    }
}

#[derive(Debug, Deserialize)]
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
        .layer(DefaultBodyLimit::max(256 * 1024))
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
    state
        .pipeline
        .predict(&observation)
        .map(Json)
        .map_err(internal_error)
}

fn bad_request(error: SdkError) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn internal_error(error: SdkError) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}
