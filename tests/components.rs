use anderion_sigint_ml::{DenseSpectrumExtractor, IqCapture, IqSample};

fn mixed_capture() -> IqCapture {
    let sample_rate = 512_000.0;
    let samples = (0..512)
        .map(|n| {
            let t = n as f64 / sample_rate;
            let a = std::f64::consts::TAU * 32_000.0 * t;
            let b = std::f64::consts::TAU * 96_000.0 * t;
            IqSample::new(
                (a.cos() + 0.6 * b.cos()) as f32,
                (a.sin() + 0.6 * b.sin()) as f32,
            )
            .unwrap()
        })
        .collect();
    IqCapture::new("mixed", 0, sample_rate, 915e6, samples).unwrap()
}

#[test]
fn dense_spectrum_returns_multiple_components() {
    let components = DenseSpectrumExtractor::new(256, 0.10)
        .unwrap()
        .extract(&mixed_capture())
        .unwrap();
    assert!(components.len() >= 2);
    assert!(
        components
            .iter()
            .any(|c| (c.center_offset_hz - 32_000.0).abs() < 3_000.0)
    );
    assert!(
        components
            .iter()
            .any(|c| (c.center_offset_hz - 96_000.0).abs() < 3_000.0)
    );
}
