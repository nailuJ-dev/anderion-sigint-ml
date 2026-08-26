use thiserror::Error;

pub type Result<T> = std::result::Result<T, SdkError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SdkError {
    #[error("feature vector must not be empty")]
    EmptyFeatures,
    #[error("value at index {index} is not finite")]
    NonFiniteValue { index: usize },
    #[error("dimension limit exceeded: actual={actual}, max={max}")]
    DimensionLimit { actual: usize, max: usize },
    #[error("dimension mismatch: expected={expected}, actual={actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("probability must be finite and in [0,1], got {0}")]
    InvalidProbability(f32),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("model is not fitted")]
    NotFitted,
    #[error("empty dataset")]
    EmptyDataset,
    #[error("artifact is too large: actual={actual}, max={max}")]
    ArtifactTooLarge { actual: usize, max: usize },
    #[error("artifact digest mismatch")]
    DigestMismatch,
    #[error("artifact schema mismatch: expected={expected}, actual={actual}")]
    SchemaMismatch { expected: u32, actual: u32 },
    #[error("artifact payload length mismatch: expected={expected}, actual={actual}")]
    PayloadLengthMismatch { expected: usize, actual: usize },
    #[error("I/O error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<std::io::Error> for SdkError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for SdkError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}
