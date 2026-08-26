use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone)]
pub struct ActiveLearningCandidate {
    id: String,
    uncertainty: f32,
    embedding: Embedding,
}

impl ActiveLearningCandidate {
    pub fn new(id: impl Into<String>, uncertainty: f32, embedding: Embedding) -> Result<Self> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(SdkError::InvalidArgument(
                "candidate id must be non-empty".into(),
            ));
        }
        if !uncertainty.is_finite() || !(0.0..=1.0).contains(&uncertainty) {
            return Err(SdkError::InvalidProbability(uncertainty));
        }
        Ok(Self {
            id,
            uncertainty,
            embedding,
        })
    }
}

pub fn select_uncertain_diverse(
    candidates: &[ActiveLearningCandidate],
    budget: usize,
    diversity_weight: f32,
) -> Result<Vec<String>> {
    if candidates.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if candidates.len() > 16_384 {
        return Err(SdkError::DimensionLimit {
            actual: candidates.len(),
            max: 16_384,
        });
    }
    if budget == 0 || budget > candidates.len() || budget > 1_024 {
        return Err(SdkError::InvalidArgument(
            "budget must be in 1..=min(candidate_count,1024)".into(),
        ));
    }
    if !diversity_weight.is_finite() || !(0.0..=1.0).contains(&diversity_weight) {
        return Err(SdkError::InvalidProbability(diversity_weight));
    }
    let dim = candidates[0].embedding.dim();
    if candidates
        .iter()
        .any(|candidate| candidate.embedding.dim() != dim)
    {
        return Err(SdkError::InvalidArgument(
            "candidate embedding dimensions must match".into(),
        ));
    }
    let mut selected_indices = Vec::with_capacity(budget);
    let first = candidates
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.uncertainty
                .total_cmp(&b.1.uncertainty)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(index, _)| index)
        .ok_or(SdkError::EmptyDataset)?;
    selected_indices.push(first);

    while selected_indices.len() < budget {
        let mut best: Option<(usize, f32)> = None;
        for (index, candidate) in candidates.iter().enumerate() {
            if selected_indices.contains(&index) {
                continue;
            }
            let min_distance = selected_indices
                .iter()
                .map(|selected| {
                    euclidean(
                        candidate.embedding.values(),
                        candidates[*selected].embedding.values(),
                    )
                })
                .fold(f32::INFINITY, f32::min);
            let diversity = min_distance / (1.0 + min_distance);
            let score =
                (1.0 - diversity_weight) * candidate.uncertainty + diversity_weight * diversity;
            match best {
                Some((best_index, best_score))
                    if best_score > score || (best_score == score && best_index < index) => {}
                _ => best = Some((index, score)),
            }
        }
        let next = best
            .map(|(index, _)| index)
            .ok_or_else(|| SdkError::InvalidArgument("unable to select candidate".into()))?;
        selected_indices.push(next);
    }
    Ok(selected_indices
        .into_iter()
        .map(|index| candidates[index].id.clone())
        .collect())
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    let sum = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let delta = f64::from(*x) - f64::from(*y);
            delta * delta
        })
        .sum::<f64>();
    sum.sqrt().min(f64::from(f32::MAX)) as f32
}
