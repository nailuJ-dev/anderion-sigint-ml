use anderion_sigint_ml::{DiagonalMetricLearner, DriftMonitor, Embedding, MeanVarianceAdapter};

#[test]
fn domain_adapter_aligns_source_mean_to_target_mean() {
    let source = vec![
        Embedding::new(vec![0.0, 0.0]).unwrap(),
        Embedding::new(vec![2.0, 2.0]).unwrap(),
    ];
    let target = vec![
        Embedding::new(vec![10.0, 20.0]).unwrap(),
        Embedding::new(vec![14.0, 24.0]).unwrap(),
    ];
    let adapter = MeanVarianceAdapter::fit(&source, &target).unwrap();
    let transformed = adapter
        .transform(&Embedding::new(vec![1.0, 1.0]).unwrap())
        .unwrap();
    assert!((transformed.values()[0] - 12.0).abs() < 1e-4);
    assert!((transformed.values()[1] - 22.0).abs() < 1e-4);
}

#[test]
fn diagonal_metric_learner_emphasizes_discriminative_dimension() {
    let positives = vec![(
        Embedding::new(vec![0.0, 0.0]).unwrap(),
        Embedding::new(vec![0.1, 2.0]).unwrap(),
    )];
    let negatives = vec![(
        Embedding::new(vec![0.0, 0.0]).unwrap(),
        Embedding::new(vec![4.0, 2.0]).unwrap(),
    )];
    let learner = DiagonalMetricLearner::fit(&positives, &negatives, 1e-4).unwrap();
    assert!(learner.weights()[0] > learner.weights()[1]);
}

#[test]
fn drift_monitor_alerts_on_large_mean_shift() {
    let baseline = vec![
        Embedding::new(vec![0.0, 0.0]).unwrap(),
        Embedding::new(vec![0.1, -0.1]).unwrap(),
        Embedding::new(vec![-0.1, 0.1]).unwrap(),
    ];
    let monitor = DriftMonitor::fit(&baseline, 3.0).unwrap();
    let current = vec![
        Embedding::new(vec![5.0, 5.0]).unwrap(),
        Embedding::new(vec![5.1, 4.9]).unwrap(),
    ];
    assert!(monitor.evaluate(&current).unwrap().drifted);
}
