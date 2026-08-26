use anderion_sigint_ml::{HashProjectionEncoder, Observation, Pipeline, PrototypeClassifier};
use std::sync::Arc;

#[test]
fn pipeline_is_deterministic_for_identical_input() {
    let encoder = HashProjectionEncoder::new(2, 4, 7).unwrap();
    let e_a = encoder.encode_features(&[1.0, 0.0]).unwrap();
    let e_b = encoder.encode_features(&[0.0, 1.0]).unwrap();
    let classifier =
        PrototypeClassifier::fit(&[(e_a, "alpha".to_string()), (e_b, "beta".to_string())]).unwrap();
    let pipeline = Pipeline::new(Arc::new(encoder), Arc::new(classifier), 0.40).unwrap();
    let obs = Observation::new("x", 1, vec![1.0, 0.0]).unwrap();

    let first = pipeline.predict(&obs).unwrap();
    let second = pipeline.predict(&obs).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.top().unwrap().label(), "alpha");
}

#[test]
fn pipeline_rejects_dimension_mismatch() {
    let encoder = HashProjectionEncoder::new(2, 2, 1).unwrap();
    let e = encoder.encode_features(&[1.0, 0.0]).unwrap();
    let classifier = PrototypeClassifier::fit(&[(e, "alpha".to_string())]).unwrap();
    let pipeline = Pipeline::new(Arc::new(encoder), Arc::new(classifier), 0.2).unwrap();
    let obs = Observation::new("x", 1, vec![1.0, 2.0, 3.0]).unwrap();

    assert!(pipeline.predict(&obs).is_err());
}
