use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{Observation, Pipeline, Result, SdkError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_runs: usize,
    pub measured_runs: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 2,
            measured_runs: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub predictions: usize,
    pub elapsed_micros: u128,
    pub mean_micros_per_prediction: f64,
    pub throughput_per_second: f64,
}

pub fn benchmark_pipeline(
    pipeline: &Pipeline,
    observations: &[Observation],
    config: BenchmarkConfig,
) -> Result<BenchmarkReport> {
    if observations.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if config.measured_runs == 0 {
        return Err(SdkError::InvalidArgument(
            "measured_runs must be positive".into(),
        ));
    }
    for _ in 0..config.warmup_runs {
        pipeline.predict_batch(observations)?;
    }
    let start = Instant::now();
    for _ in 0..config.measured_runs {
        pipeline.predict_batch(observations)?;
    }
    let elapsed = start.elapsed();
    let predictions = observations.len().saturating_mul(config.measured_runs);
    let elapsed_micros = elapsed.as_micros();
    let seconds = elapsed.as_secs_f64();
    let throughput_per_second = if seconds > 0.0 {
        predictions as f64 / seconds
    } else {
        f64::INFINITY
    };
    let mean_micros_per_prediction = if predictions > 0 {
        elapsed_micros as f64 / predictions as f64
    } else {
        0.0
    };
    Ok(BenchmarkReport {
        predictions,
        elapsed_micros,
        mean_micros_per_prediction,
        throughput_per_second,
    })
}
