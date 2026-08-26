use anderion_sigint_ml::{ArtifactManifest, ArtifactPolicy, verify_payload};

#[test]
fn artifact_verification_rejects_tampering() {
    let payload = br#"{\"model\":\"safe\"}"#;
    let manifest = ArtifactManifest::for_payload("model-1", "prototype", 1, payload).unwrap();
    let policy = ArtifactPolicy::default();
    assert!(verify_payload(&manifest, payload, &policy).is_ok());
    assert!(verify_payload(&manifest, b"tampered", &policy).is_err());
}

#[test]
fn artifact_verification_rejects_oversized_payload() {
    let payload = vec![0_u8; 16];
    let manifest = ArtifactManifest::for_payload("m", "prototype", 1, &payload).unwrap();
    let policy = ArtifactPolicy {
        max_payload_bytes: 8,
        ..ArtifactPolicy::default()
    };
    assert!(verify_payload(&manifest, &payload, &policy).is_err());
}
