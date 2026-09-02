//! Replay classification must not downgrade a semantic-consistency divergence
//! into a "decision equivalent" result.

use anderion_sigint_ml::{
    ClassScore, ConsistencyReport, DeterministicVerifier, Digest32, Embedding, Observation,
    Prediction, ReplayStatus, Result, SigintVerificationPolicy, VerificationContext,
};

fn fixtures() -> Result<(
    Observation,
    Prediction,
    VerificationContext,
    SigintVerificationPolicy,
)> {
    let observation = Observation::new("obs-1", 10, vec![1.0, 0.0])?;
    let prediction = Prediction::new(
        vec![ClassScore::new("a", 0.9)?, ClassScore::new("b", 0.1)?],
        false,
        None,
        Embedding::new(vec![1.0, 0.0])?,
        0.2,
    )?;
    let context = VerificationContext::new(
        Digest32::from_bytes(b"model"),
        Digest32::from_bytes(b"config"),
        "ontology-v1",
        "pipeline-v1",
        1,
    )?;
    let policy = SigintVerificationPolicy::new(0.2, 0.95, None, 1_000_000, false)?;
    Ok((observation, prediction, context, policy))
}

#[test]
fn identical_inputs_replay_exactly() -> Result<()> {
    let (observation, prediction, context, policy) = fixtures()?;
    let first = DeterministicVerifier::verify_sigint(
        &observation,
        &prediction,
        &context,
        &policy,
        &ConsistencyReport::valid(),
    )?;
    let second = DeterministicVerifier::verify_sigint(
        &observation,
        &prediction,
        &context,
        &policy,
        &ConsistencyReport::valid(),
    )?;
    assert_eq!(
        DeterministicVerifier::compare_replay(&first, &second),
        ReplayStatus::Exact
    );
    Ok(())
}

#[test]
fn ontology_divergence_is_not_decision_equivalent() -> Result<()> {
    let (observation, prediction, context, policy) = fixtures()?;
    let clean = DeterministicVerifier::verify_sigint(
        &observation,
        &prediction,
        &context,
        &policy,
        &ConsistencyReport::valid(),
    )?;
    let violated = DeterministicVerifier::verify_sigint(
        &observation,
        &prediction,
        &context,
        &policy,
        &ConsistencyReport::from_violation("cardinality", "conflicting assertions")?,
    )?;
    assert_eq!(
        DeterministicVerifier::compare_replay(&clean, &violated),
        ReplayStatus::NonReproducible,
        "a semantic-consistency divergence must be a hard replay failure"
    );
    Ok(())
}
