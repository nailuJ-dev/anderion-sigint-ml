use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::{IqCapture, Result, SdkError};

/// One resolved signal component of a dense spectrum capture.
///
/// `start_sample` / `end_sample` are a half-open interval `[start, end)` giving
/// the -3 dB time support of this component *after* it has been mixed to
/// baseband and low-pass filtered to its own bandwidth. They are estimated, not
/// exact edges, and they are always within the analysed capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalComponent {
    pub component_id: String,
    pub center_offset_hz: f64,
    pub bandwidth_hz: f64,
    pub start_sample: usize,
    pub end_sample: usize,
    pub relative_power_db: f32,
    pub confidence: f32,
}

/// Coarse multi-component spectrum extractor.
///
/// The capture is Hann-windowed, folded into `bins` polyphase accumulators and
/// transformed once with an FFT. This is `O(n + bins log bins)` instead of the
/// `O(n * bins)` naive DFT, and the window suppresses the spectral leakage that
/// otherwise produced spurious neighbouring "components".
///
/// Frequency convention matches [`crate::ReferenceIqFeatureExtractor`]: bins in
/// `[0, bins/2)` are positive offsets, bins in `[bins/2, bins)` are negative, so
/// `center_offset_hz` lies in `[-fs/2, +fs/2)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "DenseSpectrumExtractorWire")]
pub struct DenseSpectrumExtractor {
    bins: usize,
    relative_power_threshold: f32,
}

impl DenseSpectrumExtractor {
    pub fn new(bins: usize, relative_power_threshold: f32) -> Result<Self> {
        if !(8..=2048).contains(&bins) {
            return Err(SdkError::DimensionLimit {
                actual: bins,
                max: 2048,
            });
        }
        if !relative_power_threshold.is_finite()
            || !(0.0..=1.0).contains(&relative_power_threshold)
            || relative_power_threshold == 0.0
        {
            return Err(SdkError::InvalidArgument(
                "relative power threshold must be in (0,1]".into(),
            ));
        }
        Ok(Self {
            bins,
            relative_power_threshold,
        })
    }

    pub fn bins(&self) -> usize {
        self.bins
    }
    pub fn relative_power_threshold(&self) -> f32 {
        self.relative_power_threshold
    }

    pub fn extract(&self, capture: &IqCapture) -> Result<Vec<SignalComponent>> {
        capture.validate()?;
        let n = capture.samples().len();
        let bins = self.bins.min(n);
        let windowed = hann_windowed(capture);

        // Polyphase fold: summing the windowed capture into `bins` accumulators
        // and transforming once is the `bins`-point decimated spectrum.
        let mut folded = vec![Complex::new(0.0_f64, 0.0_f64); bins];
        for (index, value) in windowed.iter().enumerate() {
            folded[index % bins] += *value;
        }
        let mut planner = FftPlanner::<f64>::new();
        planner.plan_fft_forward(bins).process(&mut folded);

        let power: Vec<f64> = folded
            .iter()
            .map(|value| value.re.mul_add(value.re, value.im * value.im))
            .collect();
        let max_power = power.iter().copied().fold(0.0_f64, f64::max);
        if max_power <= f64::EPSILON {
            return Ok(Vec::new());
        }

        let cutoff = max_power * f64::from(self.relative_power_threshold);
        let bin_width = capture.sample_rate_hz() / bins as f64;
        let mut peaks = Vec::new();
        for bin in 0..bins {
            let left = power[(bin + bins - 1) % bins];
            let right = power[(bin + 1) % bins];
            // Strict on one side only: a flat plateau yields a single peak
            // instead of one component per plateau cell.
            if power[bin] >= cutoff && power[bin] > left && power[bin] >= right {
                peaks.push((bin, power[bin]));
            }
        }
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut out = Vec::with_capacity(peaks.len());
        for (rank, (bin, peak_power)) in peaks.into_iter().enumerate() {
            let signed_bin = if bin < bins / 2 {
                bin as isize
            } else {
                bin as isize - bins as isize
            };
            let center_offset_hz = signed_bin as f64 * bin_width;
            let bandwidth_hz = half_power_bandwidth(&power, bin, bin_width);
            let (start_sample, end_sample) = time_support(
                capture,
                center_offset_hz,
                bandwidth_hz,
                capture.sample_rate_hz(),
            );
            let relative = (peak_power / max_power).max(f64::MIN_POSITIVE);
            out.push(SignalComponent {
                component_id: format!("component-{rank:03}"),
                center_offset_hz,
                bandwidth_hz,
                start_sample,
                end_sample,
                relative_power_db: (10.0 * relative.log10()) as f32,
                confidence: relative.clamp(0.0, 1.0) as f32,
            });
        }
        out.sort_by(|a, b| a.center_offset_hz.total_cmp(&b.center_offset_hz));
        Ok(out)
    }
}

fn hann_windowed(capture: &IqCapture) -> Vec<Complex<f64>> {
    let n = capture.samples().len();
    capture
        .samples()
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let window = if n == 1 {
                1.0
            } else {
                0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / (n - 1) as f64).cos()
            };
            Complex::new(
                f64::from(sample.i()) * window,
                f64::from(sample.q()) * window,
            )
        })
        .collect()
}

/// -3 dB width around `peak_bin`, walked outward over the circular spectrum.
fn half_power_bandwidth(power: &[f64], peak_bin: usize, bin_width: f64) -> f64 {
    let bins = power.len();
    let half = power[peak_bin] * 0.5;
    let mut width = 1_usize;
    for offset in 1..=bins / 2 {
        let low = power[(peak_bin + bins - offset) % bins];
        let high = power[(peak_bin + offset) % bins];
        let mut grew = false;
        if low >= half {
            width = width.saturating_add(1);
            grew = true;
        }
        if high >= half {
            width = width.saturating_add(1);
            grew = true;
        }
        if !grew {
            break;
        }
    }
    width as f64 * bin_width
}

/// -3 dB time support of one component.
///
/// The capture is mixed down by `center_offset_hz` and smoothed by a moving
/// average whose length matches the component bandwidth; the returned interval
/// is where that envelope stays within 3 dB of its peak.
fn time_support(
    capture: &IqCapture,
    center_offset_hz: f64,
    bandwidth_hz: f64,
    sample_rate_hz: f64,
) -> (usize, usize) {
    let n = capture.samples().len();
    let window = if bandwidth_hz > 0.0 {
        ((sample_rate_hz / bandwidth_hz).round() as usize).clamp(1, n)
    } else {
        n
    };

    let mut mixed = Vec::with_capacity(n);
    for (index, sample) in capture.samples().iter().enumerate() {
        let angle = -std::f64::consts::TAU * center_offset_hz * index as f64 / sample_rate_hz;
        let (sin, cos) = angle.sin_cos();
        let i = f64::from(sample.i());
        let q = f64::from(sample.q());
        mixed.push(Complex::new(i * cos - q * sin, i * sin + q * cos));
    }

    // Running-sum moving average: one pass, no per-sample re-summation.
    let mut envelope = vec![0.0_f64; n];
    let mut accumulator = Complex::new(0.0_f64, 0.0_f64);
    for index in 0..n {
        accumulator += mixed[index];
        if index >= window {
            accumulator -= mixed[index - window];
        }
        envelope[index] = accumulator
            .re
            .mul_add(accumulator.re, accumulator.im * accumulator.im);
    }

    let peak = envelope.iter().copied().fold(0.0_f64, f64::max);
    if peak <= f64::EPSILON {
        return (0, n);
    }
    let threshold = peak * 0.5;
    let start = envelope
        .iter()
        .position(|value| *value >= threshold)
        .unwrap_or(0);
    let end = envelope
        .iter()
        .rposition(|value| *value >= threshold)
        .map_or(n, |index| index.saturating_add(1));
    // The moving average lags by up to `window` samples; pull the leading edge back.
    (
        start.saturating_sub(window.saturating_sub(1)),
        end.max(start + 1),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DenseSpectrumExtractorWire {
    bins: usize,
    relative_power_threshold: f32,
}

impl TryFrom<DenseSpectrumExtractorWire> for DenseSpectrumExtractor {
    type Error = SdkError;

    fn try_from(wire: DenseSpectrumExtractorWire) -> Result<Self> {
        Self::new(wire.bins, wire.relative_power_threshold)
    }
}
