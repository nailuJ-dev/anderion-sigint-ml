use crate::{Result, SdkError};

pub fn magnitude_prune(values: &[f32], sparsity: f32) -> Result<Vec<f32>> {
    if values.is_empty() {
        return Err(SdkError::EmptyFeatures);
    }
    if values.len() > 4_194_304 {
        return Err(SdkError::DimensionLimit {
            actual: values.len(),
            max: 4_194_304,
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SdkError::InvalidArgument(
            "pruning values must be finite".into(),
        ));
    }
    if !sparsity.is_finite() || !(0.0..1.0).contains(&sparsity) {
        return Err(SdkError::InvalidArgument(
            "sparsity must be in [0,1)".into(),
        ));
    }
    let prune_count = ((values.len() as f32) * sparsity).floor() as usize;
    let mut ranked: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.abs()))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut output = values.to_vec();
    for (index, _) in ranked.into_iter().take(prune_count) {
        output[index] = 0.0;
    }
    Ok(output)
}

pub fn sparsity(values: &[f32]) -> Result<f32> {
    if values.is_empty() {
        return Err(SdkError::EmptyFeatures);
    }
    let zeros = values.iter().filter(|value| **value == 0.0).count();
    Ok(zeros as f32 / values.len() as f32)
}
