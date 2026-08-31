use serde::{Deserialize, Serialize};
use crate::{IqCapture, Result, SdkError};

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DenseSpectrumExtractor {
    bins: usize,
    relative_power_threshold: f32,
}

impl DenseSpectrumExtractor {
    pub fn new(bins: usize, relative_power_threshold: f32) -> Result<Self> {
        if bins < 8 || bins > 2048 { return Err(SdkError::DimensionLimit { actual: bins, max: 2048 }); }
        if !relative_power_threshold.is_finite() || !(0.0..=1.0).contains(&relative_power_threshold) || relative_power_threshold == 0.0 {
            return Err(SdkError::InvalidArgument("relative power threshold must be in (0,1]".into()));
        }
        Ok(Self { bins, relative_power_threshold })
    }

    pub fn extract(&self, capture: &IqCapture) -> Result<Vec<SignalComponent>> {
        capture.validate()?;
        let n = capture.samples().len();
        let bins = self.bins.min(n);
        let mut power = vec![0.0_f64; bins];
        for (bin, dst) in power.iter_mut().enumerate() {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (index, sample) in capture.samples().iter().enumerate() {
                let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / bins as f64;
                let (sin, cos) = angle.sin_cos();
                let i = f64::from(sample.i());
                let q = f64::from(sample.q());
                re += i * cos - q * sin;
                im += i * sin + q * cos;
            }
            *dst = re.mul_add(re, im * im);
        }
        let max_power = power.iter().copied().fold(0.0_f64, f64::max);
        if max_power <= f64::EPSILON { return Ok(Vec::new()); }
        let cutoff = max_power * f64::from(self.relative_power_threshold);
        let mut peaks = Vec::new();
        for bin in 0..bins {
            let left = if bin == 0 { power[bins - 1] } else { power[bin - 1] };
            let right = if bin + 1 == bins { power[0] } else { power[bin + 1] };
            if power[bin] >= cutoff && power[bin] >= left && power[bin] >= right {
                peaks.push((bin, power[bin]));
            }
        }
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
        let bin_width = capture.sample_rate_hz() / bins as f64;
        let mut out = Vec::with_capacity(peaks.len());
        for (rank, (bin, p)) in peaks.into_iter().enumerate() {
            let signed_bin = if bin <= bins / 2 { bin as isize } else { bin as isize - bins as isize };
            let relative_db = 10.0 * (p / max_power).max(f64::MIN_POSITIVE).log10();
            out.push(SignalComponent {
                component_id: format!("component-{rank:03}"),
                center_offset_hz: signed_bin as f64 * bin_width,
                bandwidth_hz: bin_width,
                start_sample: 0,
                end_sample: n,
                relative_power_db: relative_db as f32,
                confidence: (p / max_power).clamp(0.0, 1.0) as f32,
            });
        }
        out.sort_by(|a, b| a.center_offset_hz.total_cmp(&b.center_offset_hz));
        Ok(out)
    }
}
