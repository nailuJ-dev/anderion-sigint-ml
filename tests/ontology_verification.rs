use std::sync::Arc;

use anderion_sigint_ml::{
    ConceptKind, ConsistencyReport, DeterministicVerifier, Digest32, HashProjectionEncoder,
    OntologyGraph, OntologyNode, OntologyRelation, PatternEngine, PatternEvent, PatternToken,
    Pipeline, PrototypeClassifier, RelationKind, ReplayStatus, Result, SigintVerificationPolicy,
    VerificationContext, VerifiedPipeline,
};

fn reference_pipeline() -> Result<Pipeline> {
    let encoder = HashProjectionEncoder::new(2, 2, 7)?;
    let a = encoder.encode_features(&[1.0, 0.0])?;
    let b = encoder.encode_features(&[0.0, 1.0])?;
    let classifier = PrototypeClassifier::fit(&[(a, "alpha".into()), (b, "beta".into())])?;
    Pipeline::new(Arc::new(encoder), Arc::new(classifier), 0.15)
}

fn context() -> Result<VerificationContext> {
    VerificationContext::new(
        Digest32::from_bytes(b"reference-model"),
        Digest32::from_bytes(b"reference-config"),
        "sigint-ontology-v1",
        "pipeline-v1",
        7,
    )
}

#[test]
fn ontology_detects_cardinality_contradiction_deterministically() -> Result<()> {
    let mut graph = OntologyGraph::new("sigint-ontology-v1")?;
    graph.add_node(OntologyNode::new(
        "event:1",
        ConceptKind::SignalEvent,
        None,
    )?)?;
    graph.add_node(OntologyNode::new(
        "class:a",
        ConceptKind::SignalClass,
        Some("alpha".into()),
    )?)?;
    graph.add_node(OntologyNode::new(
        "class:b",
        ConceptKind::SignalClass,
        Some("beta".into()),
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "event:1",
        RelationKind::ClassifiedAs,
        "class:a",
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "event:1",
        RelationKind::ClassifiedAs,
        "class:b",
    )?)?;

    let first = graph.validate_reference_schema();
    let second = graph.validate_reference_schema();
    assert!(!first.is_valid());
    assert_eq!(first, second);
    assert!(first.violations().iter().any(|v| v.code() == "cardinality"));
    Ok(())
}

#[test]
fn recurring_pattern_detection_is_input_order_independent() -> Result<()> {
    let burst = PatternToken::new(ConceptKind::SignalEvent, "burst", Some("cluster-a".into()))?;
    let quiet = PatternToken::new(ConceptKind::SignalEvent, "quiet", None)?;
    let events = vec![
        PatternEvent::new(40, quiet.clone()),
        PatternEvent::new(10, burst.clone()),
        PatternEvent::new(30, burst),
        PatternEvent::new(20, quiet),
    ];
    let engine = PatternEngine::default();
    let first = engine.detect_sequences(&events, 2, 2)?;
    let mut reversed = events.clone();
    reversed.reverse();
    let second = engine.detect_sequences(&reversed, 2, 2)?;
    assert_eq!(first, second);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].occurrences(), 2);
    Ok(())
}

#[test]
fn verified_pipeline_issues_repeatable_certificate_and_exact_replay() -> Result<()> {
    let policy = SigintVerificationPolicy::new(0.2, 0.95, Some(0.99), 1_000_000, true)?;
    let verified_pipeline = VerifiedPipeline::new(reference_pipeline()?, context()?, policy);
    let observation = anderion_sigint_ml::Observation::new("obs-1", 1_234, vec![1.0, 0.0])?;

    let first = verified_pipeline.predict(&observation)?;
    let second = verified_pipeline.predict(&observation)?;
    assert_eq!(first.certificate(), second.certificate());

    let (replayed, status) = verified_pipeline.replay(&observation, first.certificate())?;
    assert_eq!(status, ReplayStatus::Exact);
    assert_eq!(replayed.certificate(), first.certificate());
    Ok(())
}

#[test]
fn ontology_contradiction_forces_review_without_changing_prediction() -> Result<()> {
    let pipeline = reference_pipeline()?;
    let observation = anderion_sigint_ml::Observation::new("obs-2", 2_000, vec![0.0, 1.0])?;
    let prediction = pipeline.predict(&observation)?;
    let report =
        ConsistencyReport::from_violation("cardinality", "conflicting semantic assertions")?;
    let policy = SigintVerificationPolicy::new(0.2, 0.95, Some(0.99), 1_000_000, true)?;
    let certificate = DeterministicVerifier::verify_sigint(
        &observation,
        &prediction,
        &context()?,
        &policy,
        &report,
    )?;
    assert_eq!(
        certificate.decision(),
        anderion_sigint_ml::VerificationDecision::Review
    );
    Ok(())
}

#[test]
fn replay_rejects_context_drift() -> Result<()> {
    let policy = SigintVerificationPolicy::new(0.2, 0.95, Some(0.99), 1_000_000, true)?;
    let observation = anderion_sigint_ml::Observation::new("obs-3", 3_000, vec![1.0, 0.0])?;
    let first_pipeline = VerifiedPipeline::new(reference_pipeline()?, context()?, policy.clone());
    let first = first_pipeline.predict(&observation)?;
    let changed_context = VerificationContext::new(
        Digest32::from_bytes(b"reference-model-v2"),
        Digest32::from_bytes(b"reference-config"),
        "sigint-ontology-v1",
        "pipeline-v1",
        7,
    )?;
    let second_pipeline = VerifiedPipeline::new(reference_pipeline()?, changed_context, policy);
    let second = second_pipeline.predict(&observation)?;
    assert_eq!(
        DeterministicVerifier::compare_replay(first.certificate(), second.certificate()),
        ReplayStatus::NonReproducible
    );
    Ok(())
}

#[test]
fn digest_hex_round_trip_is_stable() -> Result<()> {
    let digest = Digest32::from_bytes(b"stable");
    assert_eq!(Digest32::from_hex(&digest.to_hex())?, digest);
    Ok(())
}
