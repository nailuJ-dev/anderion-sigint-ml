use serde::{Deserialize, Serialize};
use crate::{Embedding, HashProjectionEncoder, IqCapture, IqSample, ReferenceIqFeatureExtractor, Result, SdkError};

pub trait SpectrumEncoder {
    fn encode(&self, input: &IqCapture) -> Result<Embedding>;
    fn reconstruct_masked(&self, input: &IqCapture, mask: &[bool]) -> Result<IqCapture>;
    fn predict_next_embedding(&self, history: &[IqCapture]) -> Result<Embedding>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSpectrumEncoder {
    extractor: ReferenceIqFeatureExtractor,
    projection: HashProjectionEncoder,
}

impl ReferenceSpectrumEncoder {
    pub fn new(spectrum_bins: usize, output_dim: usize, seed: u64) -> Result<Self> {
        let extractor = ReferenceIqFeatureExtractor::new(spectrum_bins)?;
        let projection = HashProjectionEncoder::new(extractor.feature_dim(), output_dim, seed)?;
        Ok(Self { extractor, projection })
    }
}

impl SpectrumEncoder for ReferenceSpectrumEncoder {
    fn encode(&self, input: &IqCapture) -> Result<Embedding> {
        let observation = self.extractor.extract(input)?;
        self.projection.encode_features(observation.features())
    }

    fn reconstruct_masked(&self, input: &IqCapture, mask: &[bool]) -> Result<IqCapture> {
        input.validate()?;
        if mask.len() != input.samples().len() {
            return Err(SdkError::DimensionMismatch { expected: input.samples().len(), actual: mask.len() });
        }
        let mut samples = Vec::with_capacity(mask.len());
        for (sample, masked) in input.samples().iter().zip(mask) {
            samples.push(if *masked { IqSample::new(0.0, 0.0)? } else { *sample });
        }
        IqCapture::new(input.id(), input.timestamp_ms(), input.sample_rate_hz(), input.center_frequency_hz(), samples)
    }

    fn predict_next_embedding(&self, history: &[IqCapture]) -> Result<Embedding> {
        if history.is_empty() { return Err(SdkError::EmptyDataset); }
        let embeddings = history.iter().map(|capture| self.encode(capture)).collect::<Result<Vec<_>>>()?;
        let dim = embeddings[0].dim();
        if embeddings.iter().any(|embedding| embedding.dim() != dim) {
            return Err(SdkError::InvalidArgument("history embeddings must share a dimension".into()));
        }
        if embeddings.len() == 1 { return Ok(embeddings[0].clone()); }
        let last = &embeddings[embeddings.len() - 1];
        let previous = &embeddings[embeddings.len() - 2];
        let predicted = last.values().iter().zip(previous.values())
            .map(|(last, previous)| (2.0 * *last - *previous).clamp(-1.0, 1.0))
            .collect();
        Embedding::new(predicted)
    }
}
