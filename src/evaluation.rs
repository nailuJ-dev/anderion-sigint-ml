use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub macro_precision: f32,
    pub macro_recall: f32,
    pub macro_f1: f32,
    pub confusion: BTreeMap<String, BTreeMap<String, u64>>,
}

pub fn classification_metrics<T: AsRef<str>>(
    truth: &[T],
    predicted: &[T],
) -> Result<ClassificationMetrics> {
    if truth.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if truth.len() != predicted.len() {
        return Err(SdkError::DimensionMismatch {
            expected: truth.len(),
            actual: predicted.len(),
        });
    }
    let mut labels = BTreeSet::new();
    let mut confusion: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    let mut correct = 0_u64;
    for (t, p) in truth.iter().zip(predicted) {
        let t = t.as_ref().to_string();
        let p = p.as_ref().to_string();
        labels.insert(t.clone());
        labels.insert(p.clone());
        if t == p {
            correct += 1;
        }
        *confusion.entry(t).or_default().entry(p).or_insert(0) += 1;
    }
    let mut precision_sum = 0.0_f32;
    let mut recall_sum = 0.0_f32;
    let mut f1_sum = 0.0_f32;
    for label in &labels {
        let tp = confusion
            .get(label)
            .and_then(|row| row.get(label))
            .copied()
            .unwrap_or(0) as f32;
        let fp = confusion
            .iter()
            .filter(|(actual, _)| *actual != label)
            .map(|(_, row)| row.get(label).copied().unwrap_or(0))
            .sum::<u64>() as f32;
        let fn_ = confusion
            .get(label)
            .map(|row| {
                row.iter()
                    .filter(|(pred, _)| *pred != label)
                    .map(|(_, n)| *n)
                    .sum::<u64>()
            })
            .unwrap_or(0) as f32;
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
    }
    let n_labels = labels.len() as f32;
    Ok(ClassificationMetrics {
        accuracy: correct as f32 / truth.len() as f32,
        macro_precision: precision_sum / n_labels,
        macro_recall: recall_sum / n_labels,
        macro_f1: f1_sum / n_labels,
        confusion,
    })
}
