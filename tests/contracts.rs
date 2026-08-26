use anderion_sigint_ml::{ClassScore, Embedding, Observation, SdkError};

#[test]
fn observation_rejects_empty_features() {
    let error = Observation::new("obs-1", 1, vec![]).unwrap_err();
    assert!(matches!(error, SdkError::EmptyFeatures));
}

#[test]
fn observation_rejects_non_finite_features() {
    let error = Observation::new("obs-1", 1, vec![1.0, f32::NAN]).unwrap_err();
    assert!(matches!(error, SdkError::NonFiniteValue { .. }));
}

#[test]
fn embedding_rejects_excessive_dimensions() {
    let error = Embedding::with_limit(vec![0.0; 9], 8).unwrap_err();
    assert!(matches!(
        error,
        SdkError::DimensionLimit { actual: 9, max: 8 }
    ));
}

#[test]
fn class_score_rejects_invalid_probability() {
    let error = ClassScore::new("signal", 1.2).unwrap_err();
    assert!(matches!(error, SdkError::InvalidProbability(_)));
}
