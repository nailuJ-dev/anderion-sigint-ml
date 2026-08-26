use anderion_sigint_ml::{
    BenchmarkConfig, DatasetManifest, DatasetRow, Embedding, HashProjectionEncoder,
    LearnedChangePointSegmenter, MaskedContextPretrainer, Observation, OcclusionExplainer,
    Pipeline, PrototypeClassifier, SequenceClassifier, benchmark_pipeline,
};
use std::sync::Arc;

#[test]
fn self_supervised_pretrainer_reconstructs_correlated_features() {
    let samples: Vec<Embedding> = (0..20)
        .map(|i| {
            let x = i as f32 / 10.0;
            Embedding::new(vec![x, 2.0 * x]).unwrap()
        })
        .collect();
    let model = MaskedContextPretrainer::fit(&samples, 200, 0.02, 1e-4).unwrap();
    let reconstructed = model
        .reconstruct(&Embedding::new(vec![0.5, 0.0]).unwrap(), 1)
        .unwrap();
    assert!((reconstructed - 1.0).abs() < 0.25);
}

#[test]
fn learned_change_point_segmenter_finds_large_transition() {
    let training = vec![
        (
            Embedding::new(vec![0.0]).unwrap(),
            Embedding::new(vec![0.1]).unwrap(),
            false,
        ),
        (
            Embedding::new(vec![0.1]).unwrap(),
            Embedding::new(vec![0.2]).unwrap(),
            false,
        ),
        (
            Embedding::new(vec![0.0]).unwrap(),
            Embedding::new(vec![3.0]).unwrap(),
            true,
        ),
        (
            Embedding::new(vec![1.0]).unwrap(),
            Embedding::new(vec![4.0]).unwrap(),
            true,
        ),
    ];
    let model = LearnedChangePointSegmenter::fit(&training).unwrap();
    let sequence = vec![
        Embedding::new(vec![0.0]).unwrap(),
        Embedding::new(vec![0.1]).unwrap(),
        Embedding::new(vec![3.0]).unwrap(),
        Embedding::new(vec![3.1]).unwrap(),
    ];
    assert_eq!(model.boundaries(&sequence).unwrap(), vec![2]);
}

#[test]
fn sequence_classifier_returns_one_prediction_per_embedding() {
    let classifier = PrototypeClassifier::fit(&[
        (Embedding::new(vec![1.0, 0.0]).unwrap(), "a".into()),
        (Embedding::new(vec![0.0, 1.0]).unwrap(), "b".into()),
    ])
    .unwrap();
    let sequence = SequenceClassifier::new(Arc::new(classifier), 2).unwrap();
    let out = sequence
        .classify_sequence(&[
            Embedding::new(vec![0.9, 0.1]).unwrap(),
            Embedding::new(vec![0.1, 0.9]).unwrap(),
        ])
        .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0][0].label(), "a");
    assert_eq!(out[1][0].label(), "b");
}

#[test]
fn occlusion_explainer_returns_bounded_feature_importance() {
    let encoder = Arc::new(HashProjectionEncoder::new(2, 2, 9).unwrap());
    let ea = encoder.encode_features(&[1.0, 0.0]).unwrap();
    let eb = encoder.encode_features(&[0.0, 1.0]).unwrap();
    let classifier =
        Arc::new(PrototypeClassifier::fit(&[(ea, "a".into()), (eb, "b".into())]).unwrap());
    let explainer = OcclusionExplainer::new(encoder, classifier, 128).unwrap();
    let obs = Observation::new("o", 0, vec![1.0, 0.0]).unwrap();
    let explanation = explainer.explain(&obs, "a").unwrap();
    assert_eq!(explanation.importance.len(), 2);
    assert!(explanation.importance.iter().all(|v| v.is_finite()));
}

#[test]
fn dataset_manifest_is_deterministic() {
    let rows = vec![DatasetRow::new("1", "g", "a", vec![1.0, 2.0]).unwrap()];
    let a = DatasetManifest::from_rows("v1", &rows).unwrap();
    let b = DatasetManifest::from_rows("v1", &rows).unwrap();
    assert_eq!(a.sha256, b.sha256);
}

#[test]
fn pipeline_batch_and_benchmark_work() {
    let encoder = HashProjectionEncoder::new(2, 2, 7).unwrap();
    let ea = encoder.encode_features(&[1.0, 0.0]).unwrap();
    let eb = encoder.encode_features(&[0.0, 1.0]).unwrap();
    let classifier = PrototypeClassifier::fit(&[(ea, "a".into()), (eb, "b".into())]).unwrap();
    let pipeline = Pipeline::new(Arc::new(encoder), Arc::new(classifier), 0.2).unwrap();
    let inputs = vec![
        Observation::new("1", 0, vec![1.0, 0.0]).unwrap(),
        Observation::new("2", 1, vec![0.0, 1.0]).unwrap(),
    ];
    assert_eq!(pipeline.predict_batch(&inputs).unwrap().len(), 2);
    let report = benchmark_pipeline(
        &pipeline,
        &inputs,
        BenchmarkConfig {
            warmup_runs: 1,
            measured_runs: 2,
        },
    )
    .unwrap();
    assert_eq!(report.predictions, 4);
}
