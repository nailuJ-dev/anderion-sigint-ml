use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::types::validate_vector;
use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetRow {
    pub id: String,
    pub group_id: String,
    pub label: String,
    pub features: Vec<f32>,
}

impl DatasetRow {
    pub fn new(
        id: impl Into<String>,
        group_id: impl Into<String>,
        label: impl Into<String>,
        features: Vec<f32>,
    ) -> Result<Self> {
        validate_vector(&features, 65_536)?;
        let id = id.into();
        let group_id = group_id.into();
        let label = label.into();
        if id.trim().is_empty() || group_id.trim().is_empty() || label.trim().is_empty() {
            return Err(SdkError::InvalidArgument(
                "dataset id, group_id, and label must be non-empty".into(),
            ));
        }
        Ok(Self {
            id,
            group_id,
            label,
            features,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetSplit {
    pub train: Vec<DatasetRow>,
    pub test: Vec<DatasetRow>,
}

pub fn grouped_split(rows: &[DatasetRow], train_fraction: f32, seed: u64) -> Result<DatasetSplit> {
    if rows.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if !train_fraction.is_finite() || !(0.0..1.0).contains(&train_fraction) {
        return Err(SdkError::InvalidArgument(
            "train_fraction must be in (0,1)".into(),
        ));
    }
    let mut groups: BTreeMap<String, Vec<DatasetRow>> = BTreeMap::new();
    for row in rows {
        groups
            .entry(row.group_id.clone())
            .or_default()
            .push(row.clone());
    }
    if groups.len() < 2 {
        return Err(SdkError::InvalidArgument(
            "grouped split requires at least two groups".into(),
        ));
    }

    let mut ordered: Vec<(u64, String)> = groups
        .keys()
        .map(|group| (stable_hash(group.as_bytes(), seed), group.clone()))
        .collect();
    ordered.sort_by_key(|item| item.0);
    let target_train = ((rows.len() as f32) * train_fraction).round() as usize;
    let mut train_groups = BTreeSet::new();
    let mut train_count = 0_usize;
    for (_, group) in &ordered {
        if train_count >= target_train && !train_groups.is_empty() {
            break;
        }
        if let Some(items) = groups.get(group) {
            train_groups.insert(group.clone());
            train_count += items.len();
        }
    }
    if train_groups.len() == groups.len() {
        if let Some((_, last)) = ordered.last() {
            train_groups.remove(last);
        }
    }
    let mut train = Vec::new();
    let mut test = Vec::new();
    for row in rows {
        if train_groups.contains(&row.group_id) {
            train.push(row.clone());
        } else {
            test.push(row.clone());
        }
    }
    if train.is_empty() || test.is_empty() {
        return Err(SdkError::InvalidArgument(
            "split produced an empty partition".into(),
        ));
    }
    Ok(DatasetSplit { train, test })
}

fn stable_hash(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub version: String,
    pub row_count: usize,
    pub sha256: String,
}

impl DatasetManifest {
    pub fn from_rows(version: impl Into<String>, rows: &[DatasetRow]) -> Result<Self> {
        use sha2::{Digest, Sha256};
        if rows.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let version = version.into();
        if version.trim().is_empty() {
            return Err(SdkError::InvalidArgument(
                "dataset version must be non-empty".into(),
            ));
        }
        let bytes = serde_json::to_vec(rows)?;
        let digest = Sha256::digest(&bytes);
        let mut sha256 = String::with_capacity(64);
        for byte in digest {
            sha256.push_str(&format!("{byte:02x}"));
        }
        Ok(Self {
            version,
            row_count: rows.len(),
            sha256,
        })
    }
}
