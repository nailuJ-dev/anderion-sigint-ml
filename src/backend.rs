use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    External(String),
}

pub trait ComputeBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn matvec(&self, matrix: &[Vec<f32>], vector: &[f32]) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn matvec(&self, matrix: &[Vec<f32>], vector: &[f32]) -> Result<Vec<f32>> {
        if matrix.is_empty() || vector.is_empty() {
            return Err(SdkError::InvalidArgument(
                "matrix and vector must be non-empty".into(),
            ));
        }
        if matrix.iter().any(|row| row.len() != vector.len()) {
            return Err(SdkError::DimensionMismatch {
                expected: vector.len(),
                actual: matrix.iter().map(Vec::len).max().unwrap_or(0),
            });
        }
        let out = matrix
            .iter()
            .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
            .collect();
        Ok(out)
    }
}
