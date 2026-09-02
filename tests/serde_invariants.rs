//! Every type that enforces an invariant in its constructor must enforce the
//! same invariant when it is deserialized. These tests exist because a
//! `#[derive(Deserialize)]` on a struct with private fields silently bypasses
//! the constructor.

use anderion_sigint_ml::{
    ClassScore, DenseSpectrumExtractor, Embedding, IqCapture, Observation, Prediction,
    ReferenceIqFeatureExtractor, ResultCertificate,
};

#[test]
fn observation_rejects_empty_feature_vector_from_json() {
    let json = r#"{"id":"obs","timestamp_ms":1,"features":[]}"#;
    assert!(serde_json::from_str::<Observation>(json).is_err());
}

#[test]
fn observation_rejects_non_finite_features_from_json() {
    // serde_json parses to f64 and narrows to f32, so an out-of-range literal
    // becomes an infinity unless the constructor is re-run.
    let json = r#"{"id":"obs","timestamp_ms":1,"features":[1e39]}"#;
    assert!(serde_json::from_str::<Observation>(json).is_err());
}

#[test]
fn observation_rejects_empty_id_from_json() {
    let json = r#"{"id":"   ","timestamp_ms":1,"features":[1.0]}"#;
    assert!(serde_json::from_str::<Observation>(json).is_err());
}

#[test]
fn observation_rejects_unknown_fields_from_json() {
    let json = r#"{"id":"obs","timestamp_ms":1,"features":[1.0],"extra":true}"#;
    assert!(serde_json::from_str::<Observation>(json).is_err());
}

#[test]
fn embedding_rejects_empty_values_from_json() {
    assert!(serde_json::from_str::<Embedding>(r#"{"values":[]}"#).is_err());
}

#[test]
fn class_score_rejects_out_of_range_probability_from_json() {
    let json = r#"{"label":"a","probability":5.0}"#;
    assert!(serde_json::from_str::<ClassScore>(json).is_err());
}

#[test]
fn prediction_is_resorted_on_deserialization() {
    let json = r#"{
        "scores":[{"label":"low","probability":0.1},{"label":"high","probability":0.9}],
        "unknown":false,
        "anomaly_score":null,
        "embedding":{"values":[1.0,0.0]},
        "uncertainty":0.2
    }"#;
    let prediction: Prediction = serde_json::from_str(json).expect("valid prediction");
    assert_eq!(
        prediction.top().map(ClassScore::label),
        Some("high"),
        "top() must not depend on the order of the incoming JSON"
    );
}

#[test]
fn prediction_rejects_duplicate_labels_from_json() {
    let json = r#"{
        "scores":[{"label":"a","probability":0.5},{"label":"a","probability":0.5}],
        "unknown":false,
        "anomaly_score":null,
        "embedding":{"values":[1.0]},
        "uncertainty":0.2
    }"#;
    assert!(serde_json::from_str::<Prediction>(json).is_err());
}

#[test]
fn prediction_round_trips() {
    let prediction = Prediction::new(
        vec![
            ClassScore::new("a", 0.7).expect("score"),
            ClassScore::new("b", 0.3).expect("score"),
        ],
        false,
        Some(0.1),
        Embedding::new(vec![0.5, -0.5]).expect("embedding"),
        0.4,
    )
    .expect("prediction");
    let json = serde_json::to_string(&prediction).expect("serialize");
    let decoded: Prediction = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded, prediction);
}

#[test]
fn extractor_rejects_out_of_range_spectrum_bins_from_json() {
    assert!(serde_json::from_str::<ReferenceIqFeatureExtractor>(r#"{"spectrum_bins":0}"#).is_err());
    assert!(
        serde_json::from_str::<ReferenceIqFeatureExtractor>(r#"{"spectrum_bins":100000}"#).is_err()
    );
}

#[test]
fn dense_spectrum_extractor_rejects_out_of_range_bins_from_json() {
    let json = r#"{"bins":2,"relative_power_threshold":0.1}"#;
    assert!(serde_json::from_str::<DenseSpectrumExtractor>(json).is_err());
}

#[test]
fn iq_capture_rejects_short_capture_from_json() {
    let json = r#"{
        "id":"tiny","timestamp_ms":0,"sample_rate_hz":1.0,"center_frequency_hz":0.0,
        "samples":[{"i":0.0,"q":0.0},{"i":1.0,"q":0.0}]
    }"#;
    assert!(serde_json::from_str::<IqCapture>(json).is_err());
}

#[test]
fn tampered_certificate_is_rejected_on_deserialization() {
    let certificate = reference_certificate();
    let mut value: serde_json::Value =
        serde_json::to_value(&certificate).expect("certificate serializes");
    value["ontology_violation_count"] = serde_json::json!(99);
    assert!(
        serde_json::from_value::<ResultCertificate>(value).is_err(),
        "a field-edited certificate must not deserialize"
    );
}

#[test]
fn untampered_certificate_round_trips() {
    let certificate = reference_certificate();
    let json = serde_json::to_string(&certificate).expect("serialize");
    let decoded: ResultCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(decoded, certificate);
}

fn reference_certificate() -> ResultCertificate {
    use anderion_sigint_ml::{
        ConsistencyReport, DeterministicVerifier, Digest32, SigintVerificationPolicy,
        VerificationContext,
    };

    let observation = Observation::new("obs-1", 1, vec![1.0, 0.0]).expect("observation");
    let prediction = Prediction::new(
        vec![
            ClassScore::new("a", 0.8).expect("score"),
            ClassScore::new("b", 0.2).expect("score"),
        ],
        false,
        None,
        Embedding::new(vec![1.0, 0.0]).expect("embedding"),
        0.3,
    )
    .expect("prediction");
    let context = VerificationContext::new(
        Digest32::from_bytes(b"model"),
        Digest32::from_bytes(b"config"),
        "ontology-v1",
        "pipeline-v1",
        7,
    )
    .expect("context");
    let policy = SigintVerificationPolicy::new(0.2, 0.95, None, 1_000_000, true).expect("policy");
    DeterministicVerifier::verify_sigint(
        &observation,
        &prediction,
        &context,
        &policy,
        &ConsistencyReport::valid(),
    )
    .expect("certificate")
}
