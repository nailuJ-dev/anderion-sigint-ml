use anderion_sigint_ml::{
    IqCapture, IqFeatureSchema, IqSample, ReferenceIqFeatureExtractor, Result,
};

const BASE_FEATURES: usize = 12;
const BINS: usize = 16;

fn tone_capture(
    id: &str,
    normalized_frequency: f64,
    samples: usize,
    amplitude: f64,
) -> Result<IqCapture> {
    let mut iq = Vec::with_capacity(samples);
    for index in 0..samples {
        let phase = std::f64::consts::TAU * normalized_frequency * index as f64;
        iq.push(IqSample::new(
            (amplitude * phase.cos()) as f32,
            (amplitude * phase.sin()) as f32,
        )?);
    }
    IqCapture::new(id, 0, 1.0, 100.0e6, iq)
}

fn spectrum(extractor: &ReferenceIqFeatureExtractor, capture: &IqCapture) -> Result<Vec<f32>> {
    let observation = extractor.extract(capture)?;
    Ok(observation.features()[BASE_FEATURES..].to_vec())
}

fn peak_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

#[test]
fn v2_separates_positive_and_negative_complex_frequencies() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let negative = spectrum(&extractor, &tone_capture("negative", -0.30, 4096, 1.0)?)?;
    let positive = spectrum(&extractor, &tone_capture("positive", 0.30, 4096, 1.0)?)?;

    let negative_peak = peak_index(&negative);
    let positive_peak = peak_index(&positive);
    assert!(negative_peak < BINS / 2, "negative_peak={negative_peak}");
    assert!(positive_peak > BINS / 2, "positive_peak={positive_peak}");
    assert_ne!(negative_peak, positive_peak);
    Ok(())
}

#[test]
fn v2_distinguishes_low_and_high_positive_tones() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let low = spectrum(&extractor, &tone_capture("low", 0.05, 4096, 1.0)?)?;
    let high = spectrum(&extractor, &tone_capture("high", 0.40, 4096, 1.0)?)?;
    let l1: f32 = low.iter().zip(&high).map(|(a, b)| (a - b).abs()).sum();
    assert!(l1 > 1.0, "spectral L1 distance={l1}");
    Ok(())
}

#[test]
fn v2_places_dc_at_the_shifted_center() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let values = spectrum(&extractor, &tone_capture("dc", 0.0, 4096, 1.0)?)?;
    assert_eq!(peak_index(&values), BINS / 2);
    Ok(())
}

#[test]
fn v2_normalizes_nonzero_spectrum_to_unit_power() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let values = spectrum(&extractor, &tone_capture("tone", 0.177, 1024, 0.25)?)?;
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1.0e-5, "sum={sum}");
    assert!(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
    Ok(())
}

#[test]
fn v2_peak_is_stable_across_capture_lengths() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let expected = peak_index(&spectrum(
        &extractor,
        &tone_capture("n256", 0.30, 256, 1.0)?,
    )?);
    for samples in [1024, 4096] {
        let actual = peak_index(&spectrum(
            &extractor,
            &tone_capture("tone", 0.30, samples, 1.0)?,
        )?);
        assert_eq!(actual, expected, "samples={samples}");
    }
    Ok(())
}

#[test]
fn v2_spectral_shape_is_amplitude_invariant() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let a = spectrum(&extractor, &tone_capture("a", -0.22, 1024, 0.1)?)?;
    let b = spectrum(&extractor, &tone_capture("b", -0.22, 1024, 10.0)?)?;
    for (left, right) in a.iter().zip(&b) {
        assert!((left - right).abs() < 1.0e-5, "left={left} right={right}");
    }
    Ok(())
}

#[test]
fn v2_handles_maximum_capture_size_with_finite_output() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let values = spectrum(&extractor, &tone_capture("max", 0.31, 16_384, 1.0)?)?;
    assert_eq!(values.len(), BINS);
    assert!(values.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn old_serialized_extractors_load_as_explicit_legacy_v1() -> Result<()> {
    let extractor: ReferenceIqFeatureExtractor = serde_json::from_str(r#"{"spectrum_bins":16}"#)?;
    assert_eq!(extractor.schema(), IqFeatureSchema::V1NearDc);
    assert_eq!(
        ReferenceIqFeatureExtractor::new(BINS)?.schema(),
        IqFeatureSchema::V2FullBandShifted
    );
    Ok(())
}

#[test]
fn v2_zero_signal_returns_zero_spectral_tail() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let samples = (0..257)
        .map(|_| IqSample::new(0.0, 0.0))
        .collect::<Result<Vec<_>>>()?;
    let capture = IqCapture::new("zero", 0, 1.0, 100.0e6, samples)?;
    let values = spectrum(&extractor, &capture)?;
    assert!(values.iter().all(|value| *value == 0.0));
    Ok(())
}

#[test]
fn v2_fftshift_is_correct_for_odd_capture_lengths() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let negative = spectrum(&extractor, &tone_capture("odd-negative", -0.20, 1025, 1.0)?)?;
    let dc = spectrum(&extractor, &tone_capture("odd-dc", 0.0, 1025, 1.0)?)?;
    let positive = spectrum(&extractor, &tone_capture("odd-positive", 0.20, 1025, 1.0)?)?;
    assert!(peak_index(&negative) < BINS / 2);
    assert_eq!(peak_index(&dc), BINS / 2);
    assert!(peak_index(&positive) > BINS / 2);
    Ok(())
}

#[test]
fn v2_deterministic_pseudonoise_uses_most_coarse_bands() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let mut state = 0x9e37_79b9_u32;
    let mut samples = Vec::with_capacity(4096);
    for _ in 0..4096 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let i = (state as f64 / u32::MAX as f64) * 2.0 - 1.0;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let q = (state as f64 / u32::MAX as f64) * 2.0 - 1.0;
        samples.push(IqSample::new(i as f32, q as f32)?);
    }
    let capture = IqCapture::new("noise", 0, 1.0, 100.0e6, samples)?;
    let values = spectrum(&extractor, &capture)?;
    let materially_populated = values.iter().filter(|value| **value > 0.005).count();
    assert!(
        materially_populated >= 12,
        "only {materially_populated}/{BINS} coarse bands are materially populated: {values:?}"
    );
    Ok(())
}

#[test]
fn explicit_v1_matches_frozen_near_dc_reference_formula() -> Result<()> {
    let bins = 8;
    let capture = tone_capture("legacy", 0.173, 257, 0.7)?;
    let extractor = ReferenceIqFeatureExtractor::legacy_v1(bins)?;
    let actual = spectrum(&extractor, &capture)?;

    let n = capture.samples().len() as f64;
    let mut reference = Vec::with_capacity(bins);
    for bin in 0..bins {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (index, sample) in capture.samples().iter().enumerate() {
            let angle = -std::f64::consts::TAU * bin as f64 * index as f64 / n;
            let (sin, cos) = angle.sin_cos();
            let i = f64::from(sample.i());
            let q = f64::from(sample.q());
            re += i * cos - q * sin;
            im += i * sin + q * cos;
        }
        reference.push(re.mul_add(re, im * im) / (n * n));
    }
    let total: f64 = reference.iter().sum();
    let reference: Vec<f32> = reference
        .into_iter()
        .map(|value| (value / total) as f32)
        .collect();

    for (left, right) in actual.iter().zip(reference) {
        assert!((left - right).abs() < 1.0e-6, "left={left} right={right}");
    }
    Ok(())
}

#[test]
fn v2_represents_both_edges_of_the_complex_nyquist_interval() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let negative = spectrum(
        &extractor,
        &tone_capture("edge-negative", -0.49, 4096, 1.0)?,
    )?;
    let positive = spectrum(&extractor, &tone_capture("edge-positive", 0.49, 4096, 1.0)?)?;
    assert_eq!(peak_index(&negative), 0, "negative={negative:?}");
    assert_eq!(peak_index(&positive), BINS - 1, "positive={positive:?}");
    Ok(())
}

#[test]
fn newly_serialized_extractors_record_v2_schema_explicitly() -> Result<()> {
    let extractor = ReferenceIqFeatureExtractor::new(BINS)?;
    let json = serde_json::to_string(&extractor)?;
    assert!(json.contains("v2_full_band_shifted"), "json={json}");
    Ok(())
}
