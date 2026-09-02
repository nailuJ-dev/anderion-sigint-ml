use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::{Result, SdkError};

/// Read a file, refusing anything larger than `max_bytes`.
///
/// The reader is capped at `max_bytes + 1` so an oversized file is rejected
/// after one extra byte instead of being buffered in full.
pub(crate) fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
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

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Lowercase hex encoding without a per-byte allocation.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

/// Length-independent, data-independent byte comparison.
///
/// Digest comparison here guards local artifacts rather than a secret, so this
/// is defence in depth: it keeps the check from becoming a timing oracle if the
/// same helper is ever reused for a keyed digest.
pub(crate) fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut difference = 0_u8;
    for (a, b) in left.iter().zip(right) {
        difference |= a ^ b;
    }
    difference == 0
}
