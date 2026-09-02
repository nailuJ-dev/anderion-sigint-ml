use std::collections::BTreeMap;

use crate::{ClassScore, Result, SdkError};

#[derive(Debug, Clone)]
pub struct TemporalVote {
    decay: f32,
}

impl TemporalVote {
    pub fn new(decay: f32) -> Result<Self> {
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
            return Err(SdkError::InvalidProbability(decay));
        }
        Ok(Self { decay })
    }

    pub fn aggregate(&self, history: &[Vec<ClassScore>]) -> Result<Vec<ClassScore>> {
        if history.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let mut totals: BTreeMap<String, f32> = BTreeMap::new();
        let mut weight_sum = 0.0_f32;
        for (age, scores) in history.iter().rev().enumerate() {
            let weight = self.decay.powi(age as i32);
            weight_sum += weight;
            for score in scores {
                *totals.entry(score.label().to_string()).or_insert(0.0) +=
                    weight * score.probability();
            }
        }
        if weight_sum <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "temporal weights sum to zero".into(),
            ));
        }
        let raw: Vec<(String, f32)> = totals
            .into_iter()
            .map(|(label, value)| (label, value / weight_sum))
            .collect();
        normalize(raw)
    }
}

pub(crate) fn normalize(raw: Vec<(String, f32)>) -> Result<Vec<ClassScore>> {
    let sum: f32 = raw.iter().map(|(_, value)| *value).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SdkError::InvalidArgument("normalization failed".into()));
    }
    let mut out = Vec::with_capacity(raw.len());
    for (label, value) in raw {
        out.push(ClassScore::new(label, value / sum)?);
    }
    out.sort_by(|a, b| {
        b.probability()
            .total_cmp(&a.probability())
            .then_with(|| a.label().cmp(b.label()))
    });
    Ok(out)
}
