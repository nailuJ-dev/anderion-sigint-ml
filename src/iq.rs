use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::{Observation, Result, SdkError};

pub const MAX_IQ_SAMPLES: usize = 16_384;
pub const MAX_REFERENCE_SPECTRUM_BINS: usize = 64;
const BASE_FEATURES: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IqSample {
    i: f32,
    q: f32,
}

impl IqSample {
    pub fn new(i: f32, q: f32) -> Result<Self> {
        if !i.is_finite() || !q.is_finite() {
            return Err(SdkError::InvalidArgument(
                "I/Q samples must be finite".into(),
            ));
        }
        Ok(Self { i, q })
    }

    pub fn i(&self) -> f32 {
        self.i
    }
    pub fn q(&self) -> f32 {
        self.q
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IqCapture {
    id: String,
    timestamp_ms: u64,
    sample_rate_hz: f64,
    center_frequency_hz: f64,
    samples: Vec<IqSample>,
}

impl IqCapture {
    pub fn new(
        id: impl Into<String>,
        timestamp_ms: u64,
        sample_rate_hz: f64,
        center_frequency_hz: f64,
        samples: Vec<IqSample>,
    ) -> Result<Self> {
        let capture = Self {
            id: id.into(),
            timestamp_ms,
            sample_rate_hz,
            center_frequency_hz,
            samples,
        };
        capture.validate()?;
        Ok(capture)
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.id.len() > 4_096 {
            return Err(SdkError::InvalidArgument(
                "I/Q capture id must be non-empty and bounded".into(),
            ));
        }
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "sample_rate_hz must be finite and positive".into(),
            ));
        }
        if !self.center_frequency_hz.is_finite() || self.center_frequency_hz < 0.0 {
            return Err(SdkError::InvalidArgument(
                "center_frequency_hz must be finite and non-negative".into(),
            ));
        }
        if self.samples.is_empty() {
            return Err(SdkError::EmptyFeatures);
        }
        if self.samples.len() > MAX_IQ_SAMPLES {
            return Err(SdkError::DimensionLimit {
                actual: self.samples.len(),
                max: MAX_IQ_SAMPLES,
            });
        }
        for sample in &self.samples {
            IqSample::new(sample.i, sample.q)?;
        }
        Ok(())
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }
    pub fn center_frequency_hz(&self) -> f64 {
        self.center_frequency_hz
    }
    pub fn samples(&self) -> &[IqSample] {
        &self.samples
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IqFeatureSchema {
    V1NearDc,
    V2FullBandShifted,
}

fn legacy_iq_feature_schema() -> IqFeatureSchema {
    IqFeatureSchema::V1NearDc
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceIqFeatureExtractor {
    spectrum_bins: usize,
    #[serde(default = "legacy_iq_feature_schema")]
    schema: IqFeatureSchema,
}

impl Default for ReferenceIqFeatureExtractor {
    fn default() -> Self {
        Self {
            spectrum_bins: 16,
            schema: IqFeatureSchema::V2FullBandShifted,
        }
    }
}

impl ReferenceIqFeatureExtractor {
    pub fn new(spectrum_bins: usize) -> Result<Self> {
        Self::with_schema(spectrum_bins, IqFeatureSchema::V2FullBandShifted)
    }

    pub fn legacy_v1(spectrum_bins: usize) -> Result<Self> {
        Self::with_schema(spectrum_bins, IqFeatureSchema::V1NearDc)
    }

    pub fn with_schema(spectrum_bins: usize, schema: IqFeatureSchema) -> Result<Self> {
        if spectrum_bins == 0 || spectrum_bins > MAX_REFERENCE_SPECTRUM_BINS {
            return Err(SdkError::DimensionLimit {
                actual: spectrum_bins,
                max: MAX_REFERENCE_SPECTRUM_BINS,
            });
        }
        Ok(Self {
            spectrum_bins,
            schema,
        })
    }

    pub fn spectrum_bins(&self) -> usize {
        self.spectrum_bins
    }
    pub fn schema(&self) -> IqFeatureSchema {
        self.schema
    }
    pub fn feature_dim(&self) -> usize {
        BASE_FEATURES + self.spectrum_bins
    }

    pub fn extract(&self, capture: &IqCapture) -> Result<Observation> {
        capture.validate()?;
        let n = capture.samples.len() as f64;
        let mut sum_i = 0.0_f64;
        let mut sum_q = 0.0_f64;
        let mut sum_power = 0.0_f64;
        let mut sum_power_sq = 0.0_f64;
        let mut max_power = 0.0_f64;
        let mut sum_abs_i = 0.0_f64;
        let mut sum_abs_q = 0.0_f64;
        let mut sum_iq = 0.0_f64;
        let mut zero_cross_i = 0_u64;
        let mut zero_cross_q = 0_u64;
        let mut phase_step_sum = 0.0_f64;

        for (index, sample) in capture.samples.iter().enumerate() {
            let i = f64::from(sample.i);
            let q = f64::from(sample.q);
            let power = i.mul_add(i, q * q);
            sum_i += i;
            sum_q += q;
            sum_power += power;
            sum_power_sq += power * power;
            max_power = max_power.max(power);
            sum_abs_i += i.abs();
            sum_abs_q += q.abs();
            sum_iq += i * q;
            if index > 0 {
                let previous = capture.samples[index - 1];
                let pi = f64::from(previous.i);
                let pq = f64::from(previous.q);
                if (previous.i < 0.0) != (sample.i < 0.0) {
                    zero_cross_i = zero_cross_i.saturating_add(1);
                }
                if (previous.q < 0.0) != (sample.q < 0.0) {
                    zero_cross_q = zero_cross_q.saturating_add(1);
                }
                let cross = pi.mul_add(q, -(pq * i));
                let dot = pi.mul_add(i, pq * q);
                phase_step_sum += cross.atan2(dot);
            }
        }

        let mean_i = sum_i / n;
        let mean_q = sum_q / n;
        let mean_power = sum_power / n;
        let rms_amplitude = mean_power.max(0.0).sqrt();
        let variance_power = (sum_power_sq / n - mean_power * mean_power).max(0.0);
        let crest_factor = if rms_amplitude > 1e-12 {
            max_power.sqrt() / rms_amplitude
        } else {
            0.0
        };
        let transitions = capture.samples.len().saturating_sub(1).max(1) as f64;
        let phase_step_mean = phase_step_sum / transitions / std::f64::consts::PI;

        let mut features = Vec::with_capacity(self.feature_dim());
        features.extend([
            mean_i as f32,
            mean_q as f32,
            rms_amplitude as f32,
            mean_power as f32,
            variance_power.sqrt() as f32,
            crest_factor as f32,
            (sum_abs_i / n) as f32,
            (sum_abs_q / n) as f32,
            (sum_iq / n) as f32,
            (zero_cross_i as f64 / transitions) as f32,
            (zero_cross_q as f64 / transitions) as f32,
            phase_step_mean as f32,
        ]);
        features.extend(self.normalized_spectrum(capture));
        Observation::new(capture.id.clone(), capture.timestamp_ms, features)
    }

    fn normalized_spectrum(&self, capture: &IqCapture) -> Vec<f32> {
        match self.schema {
            IqFeatureSchema::V1NearDc => self.normalized_spectrum_v1(capture),
            IqFeatureSchema::V2FullBandShifted => self.normalized_spectrum_v2(capture),
        }
    }

    fn normalized_spectrum_v1(&self, capture: &IqCapture) -> Vec<f32> {
        let n = capture.samples.len() as f64;
        let mut powers = Vec::with_capacity(self.spectrum_bins);
        for bin in 0..self.spectrum_bins {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (index, sample) in capture.samples.iter().enumerate() {
                let angle = -std::f64::consts::TAU * (bin as f64) * (index as f64) / n;
                let (sin, cos) = angle.sin_cos();
                let i = f64::from(sample.i);
                let q = f64::from(sample.q);
                re += i * cos - q * sin;
                im += i * sin + q * cos;
            }
            powers.push(re.mul_add(re, im * im) / (n * n));
        }
        normalize_powers(powers, self.spectrum_bins)
    }

    fn normalized_spectrum_v2(&self, capture: &IqCapture) -> Vec<f32> {
        let n = capture.samples.len();
        let mut buffer = Vec::with_capacity(n);
        for (index, sample) in capture.samples.iter().enumerate() {
            let window = if n == 1 {
                1.0
            } else {
                0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / (n - 1) as f64).cos()
            };
            buffer.push(Complex::new(
                f64::from(sample.i) * window,
                f64::from(sample.q) * window,
            ));
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        let shift = n.div_ceil(2);
        let mut powers = vec![0.0_f64; self.spectrum_bins];
        for shifted_index in 0..n {
            let source_index = (shifted_index + shift) % n;
            // Assign the center of each shifted FFT cell to one of K equal-width
            // coarse bands over [-Fs/2, +Fs/2). The half-cell offset keeps DC in
            // the center coarse band for both even and odd capture lengths.
            let coarse_band = ((2 * shifted_index + 1) * self.spectrum_bins) / (2 * n);
            let value = buffer[source_index];
            powers[coarse_band] += value.re.mul_add(value.re, value.im * value.im);
        }
        normalize_powers(powers, self.spectrum_bins)
    }
}

fn normalize_powers(powers: Vec<f64>, spectrum_bins: usize) -> Vec<f32> {
    let total: f64 = powers.iter().sum();
    if total <= f64::EPSILON {
        return vec![0.0; spectrum_bins];
    }
    powers
        .into_iter()
        .map(|value| (value / total) as f32)
        .collect()
}
