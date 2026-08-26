use anderion_sigint_ml::{
    Classifier, Embedding, HashProjectionEncoder, OnlinePrototypeClassifier, PrototypeClassifier,
};

#[test]
fn projection_encoder_is_deterministic() {
    let encoder = HashProjectionEncoder::new(3, 8, 42).unwrap();
    let a = encoder.encode_features(&[1.0, 2.0, 3.0]).unwrap();
    let b = encoder.encode_features(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn prototype_classifier_separates_obvious_classes() {
    let classifier = PrototypeClassifier::fit(&[
        (Embedding::new(vec![1.0, 0.0]).unwrap(), "a".into()),
        (Embedding::new(vec![0.9, 0.1]).unwrap(), "a".into()),
        (Embedding::new(vec![0.0, 1.0]).unwrap(), "b".into()),
        (Embedding::new(vec![0.1, 0.9]).unwrap(), "b".into()),
    ])
    .unwrap();
    let scores = classifier
        .classify(&Embedding::new(vec![0.95, 0.05]).unwrap())
        .unwrap();
    assert_eq!(scores[0].label(), "a");
    assert!(scores[0].probability() > scores[1].probability());
}

#[test]
fn online_classifier_updates_existing_prototype() {
    let base = PrototypeClassifier::fit(&[
        (Embedding::new(vec![1.0, 0.0]).unwrap(), "a".into()),
        (Embedding::new(vec![0.0, 1.0]).unwrap(), "b".into()),
    ])
    .unwrap();
    let mut online = OnlinePrototypeClassifier::from_classifier(base);
    online
        .update("a", &Embedding::new(vec![0.8, 0.2]).unwrap())
        .unwrap();
    let scores = online
        .classify(&Embedding::new(vec![0.85, 0.15]).unwrap())
        .unwrap();
    assert_eq!(scores[0].label(), "a");
}
