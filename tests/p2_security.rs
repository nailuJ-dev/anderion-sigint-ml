use std::sync::Arc;

use anderion_sigint_ml::{
    Embedding, FoundationModel, FoundationPooler, HashProjectionEncoder, Observation,
    SymmetricQuantizer, SyntheticDataAdapter, SyntheticFeatureGenerator, VarianceAutoencoder,
    magnitude_prune,
};

#[test]
fn foundation_context_limit_is_enforced() {
    let encoder = Arc::new(HashProjectionEncoder::new(2, 2, 1).expect("encoder"));
    let model = FoundationPooler::new(encoder, 1).expect("pooler");
    let rows = vec![
        Observation::new("a", 1, vec![1.0, 0.0]).expect("observation"),
        Observation::new("b", 2, vec![0.0, 1.0]).expect("observation"),
    ];
    assert!(model.encode_window(&rows).is_err());
}

#[test]
fn invalid_quantizer_and_pruning_configuration_is_rejected() {
    assert!(SymmetricQuantizer::new(1).is_err());
    assert!(magnitude_prune(&[1.0, 2.0], 1.0).is_err());
}

#[test]
fn autoencoder_rejects_mismatched_dimensions() {
    let samples = vec![
        Embedding::new(vec![1.0, 0.0]).expect("embedding"),
        Embedding::new(vec![1.0, 0.0, 0.0]).expect("embedding"),
    ];
    assert!(VarianceAutoencoder::fit(&samples, 1).is_err());
}

#[test]
fn synthetic_generator_enforces_count_bounds() {
    let generator = SyntheticFeatureGenerator::new(4, 1).expect("generator");
    assert!(generator.generate(0).is_err());
}

#[test]
fn deserialized_autoencoder_revalidates_indices_before_use() {
    let malformed = r#"{"input_dim":2,"bottleneck_dim":1,"mean":[0.0,0.0],"selected":[99]}"#;
    let model: VarianceAutoencoder =
        serde_json::from_str(malformed).expect("json shape deserializes");
    let input = Embedding::new(vec![1.0, 0.0]).expect("embedding");
    assert!(model.encode(&input).is_err());
}
