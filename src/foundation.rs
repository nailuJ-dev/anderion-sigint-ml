use std::sync::Arc;

use crate::{Embedding, Encoder, Observation, Result, SdkError};

pub trait FoundationModel: Send + Sync {
    fn encode_window(&self, observations: &[Observation]) -> Result<Embedding>;
    fn max_context(&self) -> usize;
    fn output_dim(&self) -> usize;
}

#[derive(Clone)]
pub struct FoundationPooler {
    encoder: Arc<dyn Encoder>,
    max_context: usize,
}

impl FoundationPooler {
    pub fn new(encoder: Arc<dyn Encoder>, max_context: usize) -> Result<Self> {
        if max_context == 0 || max_context > 65_536 {
            return Err(SdkError::InvalidArgument(
                "max_context must be in 1..=65536".into(),
            ));
        }
        if encoder.output_dim() == 0 || encoder.output_dim() > 8_192 {
            return Err(SdkError::DimensionLimit {
                actual: encoder.output_dim(),
                max: 8_192,
            });
        }
        Ok(Self {
            encoder,
            max_context,
        })
    }
}

impl FoundationModel for FoundationPooler {
    fn encode_window(&self, observations: &[Observation]) -> Result<Embedding> {
        if observations.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        if observations.len() > self.max_context {
            return Err(SdkError::DimensionLimit {
                actual: observations.len(),
                max: self.max_context,
            });
        }
        let mut pooled = vec![0.0_f32; self.encoder.output_dim()];
        for observation in observations {
            if observation.features().len() != self.encoder.input_dim() {
                return Err(SdkError::DimensionMismatch {
                    expected: self.encoder.input_dim(),
                    actual: observation.features().len(),
                });
            }
            let embedding = self.encoder.encode(observation)?;
            if embedding.dim() != pooled.len() {
                return Err(SdkError::DimensionMismatch {
                    expected: pooled.len(),
                    actual: embedding.dim(),
                });
            }
            for (dst, value) in pooled.iter_mut().zip(embedding.values()) {
                *dst += *value;
            }
        }
        let count = observations.len() as f32;
        for value in &mut pooled {
            *value /= count;
        }
        Embedding::new(pooled)
    }

    fn max_context(&self) -> usize {
        self.max_context
    }
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }
}
