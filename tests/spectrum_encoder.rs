use anderion_sigint_ml::{IqCapture, IqSample, ReferenceSpectrumEncoder, SpectrumEncoder};

fn capture() -> IqCapture {
    let samples = (0..128)
        .map(|n| {
            let phase = std::f64::consts::TAU * 0.125 * n as f64;
            IqSample::new(phase.cos() as f32, phase.sin() as f32).unwrap()
        })
        .collect();
    IqCapture::new("enc", 0, 1.0e6, 2.4e9, samples).unwrap()
}

#[test]
fn reference_encoder_is_deterministic() {
    let encoder = ReferenceSpectrumEncoder::new(16, 8, 7).unwrap();
    let a = encoder.encode(&capture()).unwrap();
    let b = encoder.encode(&capture()).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.dim(), 8);
}

#[test]
fn masked_reconstruction_preserves_capture_shape() {
    let encoder = ReferenceSpectrumEncoder::new(16, 8, 7).unwrap();
    let input = capture();
    let mut mask = vec![false; input.samples().len()];
    mask[3] = true;
    let output = encoder.reconstruct_masked(&input, &mask).unwrap();
    assert_eq!(output.samples().len(), input.samples().len());
    assert_eq!(output.samples()[3].i(), 0.0);
    assert_eq!(output.samples()[3].q(), 0.0);
}
