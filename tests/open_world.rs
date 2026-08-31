use anderion_sigint_ml::{Embedding, EmitterDiscoverySession, EmitterHypothesisStatus};

#[test]
fn repeated_unknown_updates_same_provisional_emitter() {
    let mut session = EmitterDiscoverySession::new(0.90).unwrap();
    let a = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
    let b = Embedding::new(vec![0.99, 0.02, 0.0]).unwrap();
    let first = session.observe_unknown(&a).unwrap();
    let second = session.observe_unknown(&b).unwrap();
    assert_eq!(first, second);
    let hypothesis = session.hypotheses().iter().find(|h| h.id == first).unwrap();
    assert_eq!(hypothesis.support_count, 2);
    assert_eq!(hypothesis.status, EmitterHypothesisStatus::ProvisionalUnknown);
}

#[test]
fn local_enrollment_requires_confirmation_and_support() {
    let mut session = EmitterDiscoverySession::new(0.90).unwrap();
    let e = Embedding::new(vec![1.0, 0.0]).unwrap();
    let id = session.observe_unknown(&e).unwrap();
    assert!(session.enroll_local(&id, 2, true).is_err());
    session.observe_unknown(&e).unwrap();
    assert!(session.enroll_local(&id, 2, false).is_err());
    session.enroll_local(&id, 2, true).unwrap();
    let hypothesis = session.hypotheses().iter().find(|h| h.id == id).unwrap();
    assert_eq!(hypothesis.status, EmitterHypothesisStatus::EnrolledLocal);
}
