use serde::{Deserialize, Serialize};
use crate::{IqCapture, IqSample, Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiverProfile {
    pub id: String,
    pub iq_gain_imbalance: Option<f32>,
    pub iq_phase_imbalance_rad: Option<f32>,
    pub cfo_bias_hz: Option<f64>,
    pub noise_figure_db: Option<f32>,
}

impl ReceiverProfile {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), iq_gain_imbalance: None, iq_phase_imbalance_rad: None, cfo_bias_hz: None, noise_figure_db: None }
    }
    pub fn with_cfo_bias_hz(mut self, value: f64) -> Result<Self> {
        if !value.is_finite() { return Err(SdkError::InvalidArgument("CFO bias must be finite".into())); }
        self.cfo_bias_hz = Some(value); Ok(self)
    }
}

pub fn normalize_receiver_capture(capture: &IqCapture, profile: &ReceiverProfile) -> Result<IqCapture> {
    capture.validate()?;
    let gain = f64::from(profile.iq_gain_imbalance.unwrap_or(1.0));
    if !gain.is_finite() || gain.abs() <= 1e-9 { return Err(SdkError::InvalidArgument("I/Q gain correction must be finite and non-zero".into())); }
    let phase_bias = f64::from(profile.iq_phase_imbalance_rad.unwrap_or(0.0));
    if !phase_bias.is_finite() { return Err(SdkError::InvalidArgument("I/Q phase bias must be finite".into())); }
    let cfo = profile.cfo_bias_hz.unwrap_or(0.0);
    let mut samples = Vec::with_capacity(capture.samples().len());
    for (index, sample) in capture.samples().iter().enumerate() {
        let mut i = f64::from(sample.i());
        let mut q = f64::from(sample.q()) / gain;
        if phase_bias != 0.0 {
            let (sin, cos) = (-phase_bias).sin_cos();
            let next_i = i * cos - q * sin;
            let next_q = i * sin + q * cos;
            i = next_i; q = next_q;
        }
        if cfo != 0.0 {
            let angle = -std::f64::consts::TAU * cfo * index as f64 / capture.sample_rate_hz();
            let (sin, cos) = angle.sin_cos();
            let next_i = i * cos - q * sin;
            let next_q = i * sin + q * cos;
            i = next_i; q = next_q;
        }
        samples.push(IqSample::new(i as f32, q as f32)?);
    }
    IqCapture::new(capture.id(), capture.timestamp_ms(), capture.sample_rate_hz(), capture.center_frequency_hz(), samples)
}

pub fn cross_receiver_consistency(a: &IqCapture, b: &IqCapture) -> Result<f32> {
    a.validate()?; b.validate()?;
    if a.samples().len() != b.samples().len() { return Err(SdkError::DimensionMismatch { expected: a.samples().len(), actual: b.samples().len() }); }
    let mut dot = 0.0_f64; let mut aa = 0.0_f64; let mut bb = 0.0_f64;
    for (x, y) in a.samples().iter().zip(b.samples()) {
        dot += f64::from(x.i()) * f64::from(y.i()) + f64::from(x.q()) * f64::from(y.q());
        aa += f64::from(x.i()).powi(2) + f64::from(x.q()).powi(2);
        bb += f64::from(y.i()).powi(2) + f64::from(y.q()).powi(2);
    }
    if aa <= f64::EPSILON || bb <= f64::EPSILON { return Ok(0.0); }
    Ok((dot / (aa.sqrt() * bb.sqrt())).abs().clamp(0.0, 1.0) as f32)
}
