use crate::classification::squared_distance;
use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, PartialEq)]
pub struct KMeansResult {
    pub centroids: Vec<Embedding>,
    pub assignments: Vec<usize>,
    pub iterations: usize,
}

pub fn kmeans(samples: &[Embedding], k: usize, max_iterations: usize) -> Result<KMeansResult> {
    if samples.is_empty() {
        return Err(SdkError::EmptyDataset);
    }
    if k == 0 || k > samples.len() {
        return Err(SdkError::InvalidArgument(
            "k must be in 1..=sample_count".into(),
        ));
    }
    if max_iterations == 0 {
        return Err(SdkError::InvalidArgument(
            "max_iterations must be positive".into(),
        ));
    }
    let dim = samples[0].dim();
    if samples.iter().any(|sample| sample.dim() != dim) {
        return Err(SdkError::InvalidArgument(
            "all k-means samples must share a dimension".into(),
        ));
    }
    let mut centroids: Vec<Embedding> = samples.iter().take(k).cloned().collect();
    let mut assignments = vec![usize::MAX; samples.len()];
    let mut performed = 0;
    for iteration in 0..max_iterations {
        performed = iteration + 1;
        let mut changed = false;
        for (idx, sample) in samples.iter().enumerate() {
            let mut best = (f32::INFINITY, 0_usize);
            for (cluster, centroid) in centroids.iter().enumerate() {
                let distance = squared_distance(sample.values(), centroid.values())?;
                if distance < best.0 {
                    best = (distance, cluster);
                }
            }
            if assignments[idx] != best.1 {
                assignments[idx] = best.1;
                changed = true;
            }
        }
        if !changed && iteration > 0 {
            break;
        }
        let mut sums = vec![vec![0.0_f32; dim]; k];
        let mut counts = vec![0_usize; k];
        for (sample, &cluster) in samples.iter().zip(&assignments) {
            counts[cluster] += 1;
            for (dst, src) in sums[cluster].iter_mut().zip(sample.values()) {
                *dst += *src;
            }
        }
        for cluster in 0..k {
            if counts[cluster] == 0 {
                continue;
            }
            for value in &mut sums[cluster] {
                *value /= counts[cluster] as f32;
            }
            centroids[cluster] = Embedding::new(sums[cluster].clone())?;
        }
    }
    Ok(KMeansResult {
        centroids,
        assignments,
        iterations: performed,
    })
}
