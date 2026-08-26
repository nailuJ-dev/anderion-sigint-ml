use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{Encoder, Observation, Result, SdkError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeParameterProfile {
    pub parameter_count: usize,
    pub nonzero_parameters: usize,
    pub bytes_f32: usize,
    pub bytes_i8: usize,
}

pub fn profile_parameter_matrix(matrix: &[Vec<f32>]) -> Result<EdgeParameterProfile> {
    if matrix.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    let mut parameter_count = 0_usize;
    let mut nonzero_parameters = 0_usize;
    for row in matrix {
        if row.is_empty() {
            return Err(SdkError::EmptyFeatures);
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(SdkError::InvalidArgument(
                "parameters must be finite".into(),
            ));
        }
        parameter_count = parameter_count
            .checked_add(row.len())
            .ok_or_else(|| SdkError::InvalidArgument("parameter count overflow".into()))?;
        nonzero_parameters = nonzero_parameters
            .checked_add(row.iter().filter(|value| **value != 0.0).count())
            .ok_or_else(|| SdkError::InvalidArgument("nonzero count overflow".into()))?;
    }
    if parameter_count > 16_777_216 {
        return Err(SdkError::DimensionLimit {
            actual: parameter_count,
            max: 16_777_216,
        });
    }
    Ok(EdgeParameterProfile {
        parameter_count,
        nonzero_parameters,
        bytes_f32: parameter_count.saturating_mul(std::mem::size_of::<f32>()),
        bytes_i8: parameter_count,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeEncoderBenchmark {
    pub inferences: usize,
    pub elapsed_micros: u128,
    pub mean_micros: f64,
}

pub fn benchmark_encoder(
    encoder: &dyn Encoder,
    observations: &[Observation],
    runs: usize,
) -> Result<EdgeEncoderBenchmark> {
    if observations.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if runs == 0 {
        return Err(SdkError::InvalidArgument("runs must be positive".into()));
    }
    let total = observations
        .len()
        .checked_mul(runs)
        .ok_or_else(|| SdkError::InvalidArgument("benchmark inference count overflow".into()))?;
    let start = Instant::now();
    for _ in 0..runs {
        for observation in observations {
            encoder.encode(observation)?;
        }
    }
    let elapsed_micros = start.elapsed().as_micros();
    Ok(EdgeEncoderBenchmark {
        inferences: total,
        elapsed_micros,
        mean_micros: elapsed_micros as f64 / total as f64,
    })
}
