use anderion_sigint_ml::{
    IqCapture, IqSample, ReceiverProfile, cross_receiver_consistency, normalize_receiver_capture,
};

fn tone(cfo: f64) -> IqCapture {
    let sr = 256_000.0;
    let samples = (0..256)
        .map(|n| {
            let phase = std::f64::consts::TAU * (40_000.0 + cfo) * n as f64 / sr;
            IqSample::new(phase.cos() as f32, phase.sin() as f32).unwrap()
        })
        .collect();
    IqCapture::new("tone", 0, sr, 433.92e6, samples).unwrap()
}

#[test]
fn receiver_profile_removes_declared_cfo_bias() {
    let reference = tone(0.0);
    let shifted = tone(2_500.0);
    let profile = ReceiverProfile::new("rx-b")
        .with_cfo_bias_hz(2_500.0)
        .unwrap();
    let corrected = normalize_receiver_capture(&shifted, &profile).unwrap();
    let score = cross_receiver_consistency(&reference, &corrected).unwrap();
    assert!(score > 0.99);
}
