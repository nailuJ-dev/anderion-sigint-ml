use anderion_sigint_ml::{
    GoldenSigintModel, GoldenSigintSample, IqCapture, IqSample, ReferenceIqFeatureExtractor,
    ReplayStatus,
};

fn tone_capture(id: &str, phase: f32) -> IqCapture {
    let samples = (0..64)
        .map(|index| {
            let angle = 2.0_f32 * std::f32::consts::PI * (index as f32) / 16.0 + phase;
            IqSample::new(angle.cos(), angle.sin()).expect("finite fixture")
        })
        .collect();
    IqCapture::new(id, 1_000, 1_000_000.0, 2_450_000_000.0, samples).expect("valid fixture")
}

#[test]
fn iq_feature_extraction_is_deterministic_and_fixed_dimension() {
    let extractor = ReferenceIqFeatureExtractor::new(8).expect("valid extractor");
    let capture = tone_capture("tone", 0.0);
    let first = extractor.extract(&capture).expect("extracts");
    let second = extractor.extract(&capture).expect("extracts");
    assert_eq!(first.features(), second.features());
    assert_eq!(first.features().len(), extractor.feature_dim());
}

#[test]
fn golden_model_replays_exactly_for_identical_capture() {
    let training = vec![
        GoldenSigintSample::new("tone", tone_capture("tone-a", 0.0)).expect("sample"),
        GoldenSigintSample::new("tone", tone_capture("tone-b", 0.2)).expect("sample"),
        GoldenSigintSample::new(
            "quiet",
            IqCapture::new(
                "quiet-a",
                1_001,
                1_000_000.0,
                2_450_000_000.0,
                vec![IqSample::new(0.01, 0.0).expect("sample"); 64],
            )
            .expect("capture"),
        )
        .expect("sample"),
        GoldenSigintSample::new(
            "quiet",
            IqCapture::new(
                "quiet-b",
                1_002,
                1_000_000.0,
                2_450_000_000.0,
                vec![IqSample::new(0.02, 0.0).expect("sample"); 64],
            )
            .expect("capture"),
        )
        .expect("sample"),
    ];
    let model = GoldenSigintModel::fit(&training, 7).expect("fit");
    let capture = tone_capture("evaluation", 0.1);
    let report = model.infer(&capture, Some("tone")).expect("infer");
    let (_, replay) = model
        .replay(&capture, report.certificate())
        .expect("replay");
    assert_eq!(replay, ReplayStatus::Exact);
    assert_eq!(report.expected_label(), Some("tone"));
}
