use anderion_sigint_ml::{
    AnomalyDetector, ClassScore, DiagonalGaussianAnomalyDetector, Embedding, NearestPrototypeOod,
    TemperatureScaler, normalized_entropy,
};

#[test]
fn anomaly_detector_ranks_far_sample_higher() {
    let detector = DiagonalGaussianAnomalyDetector::fit(&[
        Embedding::new(vec![0.0, 0.0]).unwrap(),
        Embedding::new(vec![0.1, -0.1]).unwrap(),
        Embedding::new(vec![-0.1, 0.1]).unwrap(),
    ])
    .unwrap();
    let near = detector
        .anomaly_score(&Embedding::new(vec![0.05, 0.0]).unwrap())
        .unwrap();
    let far = detector
        .anomaly_score(&Embedding::new(vec![10.0, 10.0]).unwrap())
        .unwrap();
    assert!(far > near);
}

#[test]
fn open_set_rejects_distant_embedding() {
    let ood = NearestPrototypeOod::new(vec![Embedding::new(vec![1.0, 0.0]).unwrap()], 0.5).unwrap();
    assert!(
        !ood.is_unknown(&Embedding::new(vec![0.9, 0.1]).unwrap())
            .unwrap()
    );
    assert!(
        ood.is_unknown(&Embedding::new(vec![-1.0, 0.0]).unwrap())
            .unwrap()
    );
}

#[test]
fn normalized_entropy_is_bounded() {
    let scores = vec![
        ClassScore::new("a", 0.5).unwrap(),
        ClassScore::new("b", 0.5).unwrap(),
    ];
    let entropy = normalized_entropy(&scores).unwrap();
    assert!((0.0..=1.0).contains(&entropy));
    assert!((entropy - 1.0).abs() < 1e-5);
}

#[test]
fn temperature_scaler_preserves_probability_sum() {
    let scaler = TemperatureScaler::new(1.5).unwrap();
    let scores = vec![
        ClassScore::new("a", 0.8).unwrap(),
        ClassScore::new("b", 0.2).unwrap(),
    ];
    let calibrated = scaler.calibrate_scores(&scores).unwrap();
    let sum: f32 = calibrated.iter().map(|s| s.probability()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
}
