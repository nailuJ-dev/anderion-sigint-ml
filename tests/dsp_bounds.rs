//! Bounds and conventions of the reference DSP front end.

use anderion_sigint_ml::{
    DenseSpectrumExtractor, IqCapture, IqSample, MIN_IQ_SAMPLES, ReferenceIqFeatureExtractor,
    Result,
};

fn tone(id: &str, normalized_frequency: f64, samples: usize) -> Result<IqCapture> {
    let iq = (0..samples)
        .map(|index| {
            let phase = std::f64::consts::TAU * normalized_frequency * index as f64;
            IqSample::new(phase.cos() as f32, phase.sin() as f32)
        })
        .collect::<Result<Vec<_>>>()?;
    IqCapture::new(id, 0, 1.0, 100.0e6, iq)
}

#[test]
fn captures_below_the_window_floor_are_rejected() {
    // A symmetric Hann window zeroes the first and last tap: at n == 2 the whole
    // capture disappears and the spectral tail is silently all-zero.
    for length in 1..MIN_IQ_SAMPLES {
        let samples = vec![IqSample::new(1.0, 0.0).expect("sample"); length];
        assert!(
            IqCapture::new("short", 0, 1.0, 0.0, samples).is_err(),
            "capture of {length} samples must be rejected"
        );
    }
    let samples = vec![IqSample::new(1.0, 0.0).expect("sample"); MIN_IQ_SAMPLES];
    assert!(IqCapture::new("floor", 0, 1.0, 0.0, samples).is_ok());
}

#[test]
fn more_coarse_bands_than_fft_cells_is_rejected() {
    let extractor = ReferenceIqFeatureExtractor::new(64).expect("extractor");
    let capture = tone("short", 0.1, 32).expect("capture");
    assert!(
        extractor.extract(&capture).is_err(),
        "64 coarse bands over 32 FFT cells cannot keep DC in the centre band"
    );
}

#[test]
fn dense_spectrum_uses_the_same_frequency_convention_as_the_extractor() {
    // Positive baseband tone -> positive offset; negative tone -> negative offset.
    let sample_rate = 512_000.0;
    let build = |frequency_hz: f64| {
        let samples = (0..512)
            .map(|index| {
                let t = index as f64 / sample_rate;
                let angle = std::f64::consts::TAU * frequency_hz * t;
                IqSample::new(angle.cos() as f32, angle.sin() as f32).expect("sample")
            })
            .collect();
        IqCapture::new("tone", 0, sample_rate, 915e6, samples).expect("capture")
    };
    let extractor = DenseSpectrumExtractor::new(256, 0.10).expect("extractor");

    let positive = extractor.extract(&build(96_000.0)).expect("components");
    assert!(
        positive
            .iter()
            .any(|component| (component.center_offset_hz - 96_000.0).abs() < 3_000.0),
        "positive tone missing: {positive:?}"
    );

    let negative = extractor.extract(&build(-96_000.0)).expect("components");
    assert!(
        negative
            .iter()
            .any(|component| (component.center_offset_hz + 96_000.0).abs() < 3_000.0),
        "negative tone missing: {negative:?}"
    );
}

#[test]
fn dense_spectrum_reports_a_bounded_time_support() {
    let sample_rate = 512_000.0;
    let samples = (0..512)
        .map(|index| {
            let t = index as f64 / sample_rate;
            let angle = std::f64::consts::TAU * 32_000.0 * t;
            IqSample::new(angle.cos() as f32, angle.sin() as f32).expect("sample")
        })
        .collect();
    let capture = IqCapture::new("tone", 0, sample_rate, 915e6, samples).expect("capture");
    let components = DenseSpectrumExtractor::new(256, 0.10)
        .expect("extractor")
        .extract(&capture)
        .expect("components");
    for component in &components {
        assert!(component.start_sample < component.end_sample);
        assert!(component.end_sample <= 512);
        assert!(component.bandwidth_hz > 0.0);
    }
}
