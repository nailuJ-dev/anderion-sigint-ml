use std::sync::Arc;

use anderion_sigint_ml::Classifier;
use anderion_sigint_ml::{
    ActiveLearningCandidate, ContrastiveProjector, DistilledPrototypeClassifier,
    EdgeParameterProfile, Embedding, EmbeddingZeroShotClassifier, FoundationModel,
    FoundationPooler, HashProjectionEncoder, Observation, QuantizationAwarePrototypeClassifier,
    SoftLabelSample, SymmetricQuantizer, SyntheticDataAdapter, SyntheticFeatureGenerator,
    TemporalSelfAttentionEncoder, VarianceAutoencoder, magnitude_prune, profile_parameter_matrix,
    select_uncertain_diverse,
};

fn embedding(values: &[f32]) -> Embedding {
    Embedding::new(values.to_vec()).expect("test embedding is valid")
}

#[test]
fn foundation_pooler_encodes_a_bounded_context() {
    let encoder = Arc::new(HashProjectionEncoder::new(3, 2, 7).expect("valid encoder"));
    let model = FoundationPooler::new(encoder, 4).expect("valid foundation pooler");
    let observations = vec![
        Observation::new("a", 1, vec![1.0, 0.0, 0.0]).expect("valid observation"),
        Observation::new("b", 2, vec![0.0, 1.0, 0.0]).expect("valid observation"),
    ];
    let output = model.encode_window(&observations).expect("window encodes");
    assert_eq!(output.dim(), 2);
}

#[test]
fn temporal_attention_preserves_embedding_dimension() {
    let attention = TemporalSelfAttentionEncoder::new(0.7).expect("valid attention");
    let output = attention
        .aggregate(&[embedding(&[1.0, 0.0]), embedding(&[0.8, 0.2])])
        .expect("attention succeeds");
    assert_eq!(output.dim(), 2);
}

#[test]
fn variance_autoencoder_round_trips_through_bottleneck() {
    let model = VarianceAutoencoder::fit(
        &[
            embedding(&[1.0, 0.0, 0.0]),
            embedding(&[2.0, 0.0, 1.0]),
            embedding(&[3.0, 0.0, 2.0]),
        ],
        2,
    )
    .expect("autoencoder fits");
    let latent = model.encode(&embedding(&[2.0, 0.0, 1.0])).expect("encode");
    let reconstructed = model.decode(&latent).expect("decode");
    assert_eq!(latent.dim(), 2);
    assert_eq!(reconstructed.dim(), 3);
}

#[test]
fn contrastive_projector_learns_finite_scaling() {
    let model = ContrastiveProjector::fit(&[
        (embedding(&[1.0, 0.0]), embedding(&[1.1, 0.0]), true),
        (embedding(&[1.0, 0.0]), embedding(&[-1.0, 0.0]), false),
    ])
    .expect("contrastive fit");
    let transformed = model.transform(&embedding(&[0.5, 0.5])).expect("transform");
    assert_eq!(transformed.dim(), 2);
    assert!(transformed.values().iter().all(|value| value.is_finite()));
}

#[test]
fn active_learning_combines_uncertainty_and_diversity() {
    let candidates = vec![
        ActiveLearningCandidate::new("a", 0.9, embedding(&[1.0, 0.0])).expect("candidate"),
        ActiveLearningCandidate::new("b", 0.8, embedding(&[0.9, 0.1])).expect("candidate"),
        ActiveLearningCandidate::new("c", 0.7, embedding(&[-1.0, 0.0])).expect("candidate"),
    ];
    let selected = select_uncertain_diverse(&candidates, 2, 0.5).expect("selection");
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0], "a");
}

#[test]
fn zero_shot_classifier_uses_user_supplied_prototypes() {
    let classifier = EmbeddingZeroShotClassifier::new(
        vec![
            ("alpha".to_string(), embedding(&[1.0, 0.0])),
            ("beta".to_string(), embedding(&[0.0, 1.0])),
        ],
        0.2,
    )
    .expect("zero-shot classifier");
    let scores = classifier
        .classify(&embedding(&[0.9, 0.1]))
        .expect("classification");
    assert_eq!(scores.first().map(|score| score.label()), Some("alpha"));
}

#[test]
fn distillation_builds_a_classifier_from_soft_labels() {
    let samples = vec![
        SoftLabelSample::new(
            embedding(&[1.0, 0.0]),
            vec![
                anderion_sigint_ml::ClassScore::new("alpha", 0.9).expect("score"),
                anderion_sigint_ml::ClassScore::new("beta", 0.1).expect("score"),
            ],
        )
        .expect("soft label"),
        SoftLabelSample::new(
            embedding(&[0.0, 1.0]),
            vec![
                anderion_sigint_ml::ClassScore::new("alpha", 0.1).expect("score"),
                anderion_sigint_ml::ClassScore::new("beta", 0.9).expect("score"),
            ],
        )
        .expect("soft label"),
    ];
    let student = DistilledPrototypeClassifier::fit(&samples, 1.0).expect("distillation");
    let scores = student
        .classify(&embedding(&[0.95, 0.05]))
        .expect("student predicts");
    assert_eq!(scores.first().map(|score| score.label()), Some("alpha"));
}

#[test]
fn symmetric_quantization_round_trip_is_bounded() {
    let quantizer = SymmetricQuantizer::new(8).expect("quantizer");
    let original = embedding(&[-1.0, -0.25, 0.5, 1.0]);
    let quantized = quantizer.quantize(&original).expect("quantize");
    let restored = quantizer.dequantize(&quantized).expect("dequantize");
    assert_eq!(restored.dim(), original.dim());
    assert!(restored.values().iter().all(|value| value.is_finite()));
}

#[test]
fn magnitude_pruning_and_edge_profile_report_compression() {
    let pruned = magnitude_prune(&[0.01, 3.0, -2.0, 0.02], 0.5).expect("prune");
    assert_eq!(pruned.iter().filter(|value| **value == 0.0).count(), 2);
    let profile: EdgeParameterProfile = profile_parameter_matrix(&[pruned]).expect("profile");
    assert_eq!(profile.parameter_count, 4);
    assert_eq!(profile.nonzero_parameters, 2);
    assert!(profile.bytes_f32 > profile.bytes_i8);
}

#[test]
fn synthetic_adapter_generates_generic_bounded_observations() {
    let generator = SyntheticFeatureGenerator::new(4, 99).expect("generator");
    let rows = generator.generate(3).expect("generate");
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.features().len() == 4));
}

#[test]
fn temporal_transformer_encoder_returns_normalized_context_embedding() {
    let model =
        anderion_sigint_ml::TemporalTransformerEncoder::new(1.0, 0.25).expect("transformer");
    let output = model
        .aggregate(&[embedding(&[1.0, 0.0]), embedding(&[0.5, 0.5])])
        .expect("aggregate");
    assert_eq!(output.dim(), 2);
    assert!(output.values().iter().all(|value| value.is_finite()));
}

#[test]
fn quantization_aware_reference_classifier_trains_on_fake_quantized_embeddings() {
    let samples = vec![
        (embedding(&[1.0, 0.0]), "alpha".to_string()),
        (embedding(&[0.0, 1.0]), "beta".to_string()),
    ];
    let classifier =
        QuantizationAwarePrototypeClassifier::fit(&samples, 8).expect("qat classifier");
    let scores = classifier
        .classify(&embedding(&[0.9, 0.1]))
        .expect("classification");
    assert_eq!(scores.first().map(|score| score.label()), Some("alpha"));
}
