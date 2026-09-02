use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityHit {
    pub id: String,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
pub struct SimilarityIndex {
    dimension: usize,
    entries: Vec<(String, Embedding)>,
    max_entries: usize,
}

impl SimilarityIndex {
    pub fn new(dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(SdkError::InvalidArgument(
                "similarity dimension must be positive".into(),
            ));
        }
        Ok(Self {
            dimension,
            entries: Vec::new(),
            max_entries: 1_000_000,
        })
    }

    pub fn with_capacity_limit(mut self, max_entries: usize) -> Result<Self> {
        if max_entries == 0 {
            return Err(SdkError::InvalidArgument(
                "max_entries must be positive".into(),
            ));
        }
        self.max_entries = max_entries;
        Ok(self)
    }

    pub fn insert(&mut self, id: impl Into<String>, embedding: Embedding) -> Result<()> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        if self.entries.len() >= self.max_entries {
            return Err(SdkError::DimensionLimit {
                actual: self.entries.len() + 1,
                max: self.max_entries,
            });
        }
        self.entries.push((id.into(), embedding));
        Ok(())
    }

    pub fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SimilarityHit>> {
        if query.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dim(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut hits = self
            .entries
            .iter()
            .map(|(id, embedding)| {
                cosine_similarity(query.values(), embedding.values()).map(|similarity| {
                    SimilarityHit {
                        id: id.clone(),
                        similarity,
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        hits.sort_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(k.min(hits.len()));
        Ok(hits)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(SdkError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return Ok(0.0);
    }
    Ok((dot / (na * nb)).clamp(-1.0, 1.0))
}
