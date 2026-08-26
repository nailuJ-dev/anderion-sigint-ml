use crate::{ClassScore, Result, SdkError};

pub fn normalized_entropy(scores: &[ClassScore]) -> Result<f32> {
    if scores.is_empty() {
        return Err(SdkError::InvalidArgument(
            "entropy requires at least one class".into(),
        ));
    }
    if scores.len() == 1 {
        return Ok(0.0);
    }
    let sum: f32 = scores.iter().map(ClassScore::probability).sum();
    if (sum - 1.0).abs() > 1e-3 {
        return Err(SdkError::InvalidArgument(
            "class probabilities must sum to one".into(),
        ));
    }
    let entropy = scores
        .iter()
        .map(|score| {
            let p = score.probability().max(f32::MIN_POSITIVE);
            -p * p.ln()
        })
        .sum::<f32>();
    Ok((entropy / (scores.len() as f32).ln()).clamp(0.0, 1.0))
}
