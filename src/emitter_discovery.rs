use serde::{Deserialize, Serialize};
use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmitterHypothesisStatus { Known, ProvisionalUnknown, EnrolledLocal, Abstain }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmitterHypothesis {
    pub id: String,
    pub support_count: usize,
    pub prototype: Embedding,
    pub confidence: f32,
    pub status: EmitterHypothesisStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitterDiscoverySession {
    similarity_threshold: f32,
    next_unknown: u64,
    hypotheses: Vec<EmitterHypothesis>,
}

impl EmitterDiscoverySession {
    pub fn new(similarity_threshold: f32) -> Result<Self> {
        if !similarity_threshold.is_finite() || !(0.0..=1.0).contains(&similarity_threshold) {
            return Err(SdkError::InvalidProbability(similarity_threshold));
        }
        Ok(Self { similarity_threshold, next_unknown: 1, hypotheses: Vec::new() })
    }

    pub fn hypotheses(&self) -> &[EmitterHypothesis] { &self.hypotheses }

    pub fn observe_unknown(&mut self, embedding: &Embedding) -> Result<String> {
        if embedding.dim() == 0 { return Err(SdkError::EmptyFeatures); }
        let mut best: Option<(usize, f32)> = None;
        for (index, hypothesis) in self.hypotheses.iter().enumerate() {
            if hypothesis.prototype.dim() != embedding.dim() { continue; }
            let similarity = cosine(hypothesis.prototype.values(), embedding.values())?;
            if similarity >= self.similarity_threshold && best.map_or(true, |(_, score)| similarity > score) {
                best = Some((index, similarity));
            }
        }
        if let Some((index, similarity)) = best {
            let hypothesis = &mut self.hypotheses[index];
            let next_count = hypothesis.support_count.saturating_add(1);
            let mut values = hypothesis.prototype.values().to_vec();
            for (dst, src) in values.iter_mut().zip(embedding.values()) {
                *dst += (*src - *dst) / next_count as f32;
            }
            hypothesis.prototype = Embedding::new(values)?;
            hypothesis.support_count = next_count;
            hypothesis.confidence = similarity;
            return Ok(hypothesis.id.clone());
        }
        let id = format!("rf-unknown-{:04}", self.next_unknown);
        self.next_unknown = self.next_unknown.saturating_add(1);
        self.hypotheses.push(EmitterHypothesis {
            id: id.clone(), support_count: 1, prototype: embedding.clone(), confidence: 0.5,
            status: EmitterHypothesisStatus::ProvisionalUnknown,
        });
        Ok(id)
    }

    pub fn enroll_local(&mut self, id: &str, min_support: usize, operator_confirmed: bool) -> Result<()> {
        if min_support == 0 { return Err(SdkError::InvalidArgument("minimum support must be positive".into())); }
        if !operator_confirmed { return Err(SdkError::InvalidArgument("local enrollment requires explicit operator confirmation".into())); }
        let hypothesis = self.hypotheses.iter_mut().find(|hypothesis| hypothesis.id == id)
            .ok_or_else(|| SdkError::InvalidArgument("emitter hypothesis not found".into()))?;
        if hypothesis.support_count < min_support {
            return Err(SdkError::InvalidArgument("insufficient evidence for local enrollment".into()));
        }
        hypothesis.status = EmitterHypothesisStatus::EnrolledLocal;
        hypothesis.confidence = hypothesis.confidence.max(0.75);
        Ok(())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() { return Err(SdkError::DimensionMismatch { expected: a.len(), actual: b.len() }); }
    let mut dot = 0.0_f32; let mut aa = 0.0_f32; let mut bb = 0.0_f32;
    for (x, y) in a.iter().zip(b) { dot += x * y; aa += x * x; bb += y * y; }
    if aa <= f32::EPSILON || bb <= f32::EPSILON { return Ok(0.0); }
    Ok((dot / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0))
}
