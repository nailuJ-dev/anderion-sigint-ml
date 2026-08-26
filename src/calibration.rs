use serde::{Deserialize, Serialize};

use crate::{Calibrator, ClassScore, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScaler {
    temperature: f32,
}

impl TemperatureScaler {
    pub fn new(temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "temperature must be finite and positive".into(),
            ));
        }
        Ok(Self { temperature })
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn calibrate_scores(&self, scores: &[ClassScore]) -> Result<Vec<ClassScore>> {
        self.calibrate(scores)
    }

    pub fn fit(calibration: &[(Vec<ClassScore>, String)]) -> Result<Self> {
        if calibration.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let candidates = [0.5_f32, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0];
        let mut best = (f32::INFINITY, 1.0_f32);
        for temperature in candidates {
            let scaler = Self::new(temperature)?;
            let mut nll = 0.0_f32;
            for (scores, label) in calibration {
                let calibrated = scaler.calibrate(scores)?;
                let probability = calibrated
                    .iter()
                    .find(|score| score.label() == label)
                    .map(ClassScore::probability)
                    .unwrap_or(f32::MIN_POSITIVE)
                    .max(f32::MIN_POSITIVE);
                nll -= probability.ln();
            }
            nll /= calibration.len() as f32;
            if nll < best.0 {
                best = (nll, temperature);
            }
        }
        Self::new(best.1)
    }
}

impl Calibrator for TemperatureScaler {
    fn calibrate(&self, scores: &[ClassScore]) -> Result<Vec<ClassScore>> {
        if scores.is_empty() {
            return Err(SdkError::InvalidArgument(
                "calibration requires scores".into(),
            ));
        }
        let inv_t = 1.0 / self.temperature;
        let transformed: Vec<f32> = scores
            .iter()
            .map(|score| score.probability().max(f32::MIN_POSITIVE).powf(inv_t))
            .collect();
        let sum: f32 = transformed.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "calibration normalization failed".into(),
            ));
        }
        let mut out = Vec::with_capacity(scores.len());
        for (score, value) in scores.iter().zip(transformed) {
            out.push(ClassScore::new(score.label().to_string(), value / sum)?);
        }
        out.sort_by(|a, b| b.probability().total_cmp(&a.probability()));
        Ok(out)
    }
}
