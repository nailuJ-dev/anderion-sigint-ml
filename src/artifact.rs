use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub model_type: String,
    pub payload_len: usize,
    pub payload_sha256: String,
}

impl ArtifactManifest {
    pub fn for_payload(
        model_id: impl Into<String>,
        model_type: impl Into<String>,
        schema_version: u32,
        payload: &[u8],
    ) -> Result<Self> {
        let model_id = model_id.into();
        let model_type = model_type.into();
        if model_id.trim().is_empty() || model_type.trim().is_empty() {
            return Err(SdkError::InvalidArgument(
                "model_id and model_type must be non-empty".into(),
            ));
        }
        Ok(Self {
            schema_version,
            model_id,
            model_type,
            payload_len: payload.len(),
            payload_sha256: sha256_hex(payload),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactPolicy {
    pub expected_schema_version: u32,
    pub max_payload_bytes: usize,
    pub allowed_model_types: Vec<String>,
}

impl Default for ArtifactPolicy {
    fn default() -> Self {
        Self {
            expected_schema_version: 1,
            max_payload_bytes: 128 * 1024 * 1024,
            allowed_model_types: vec!["prototype".into(), "reference_bundle".into()],
        }
    }
}

pub fn verify_payload(
    manifest: &ArtifactManifest,
    payload: &[u8],
    policy: &ArtifactPolicy,
) -> Result<()> {
    if manifest.schema_version != policy.expected_schema_version {
        return Err(SdkError::SchemaMismatch {
            expected: policy.expected_schema_version,
            actual: manifest.schema_version,
        });
    }
    if payload.len() > policy.max_payload_bytes {
        return Err(SdkError::ArtifactTooLarge {
            actual: payload.len(),
            max: policy.max_payload_bytes,
        });
    }
    if payload.len() != manifest.payload_len {
        return Err(SdkError::PayloadLengthMismatch {
            expected: manifest.payload_len,
            actual: payload.len(),
        });
    }
    if !policy
        .allowed_model_types
        .iter()
        .any(|item| item == &manifest.model_type)
    {
        return Err(SdkError::InvalidArgument(format!(
            "model type '{}' is not allowed",
            manifest.model_type
        )));
    }
    if sha256_hex(payload) != manifest.payload_sha256.to_ascii_lowercase() {
        return Err(SdkError::DigestMismatch);
    }
    Ok(())
}

pub fn load_verified_payload(
    manifest_path: impl AsRef<Path>,
    payload_path: impl AsRef<Path>,
    policy: &ArtifactPolicy,
) -> Result<(ArtifactManifest, Vec<u8>)> {
    let manifest_bytes = read_bounded(manifest_path.as_ref(), 1024 * 1024)?;
    let manifest: ArtifactManifest = serde_json::from_slice(&manifest_bytes)?;
    let payload = read_bounded(payload_path.as_ref(), policy.max_payload_bytes)?;
    verify_payload(&manifest, &payload, policy)?;
    Ok((manifest, payload))
}

fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let limit = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut reader = file.take(limit);
    let mut bytes = Vec::with_capacity(max_bytes.min(1024 * 1024));
    reader.read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(SdkError::ArtifactTooLarge {
            actual: bytes.len(),
            max: max_bytes,
        });
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
