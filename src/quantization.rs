use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedEmbedding {
    pub values: Vec<i8>,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SymmetricQuantizer {
    bits: u8,
}

impl SymmetricQuantizer {
    pub fn new(bits: u8) -> Result<Self> {
        if !(2..=8).contains(&bits) {
            return Err(SdkError::InvalidArgument(
                "quantizer bits must be in 2..=8".into(),
            ));
        }
        Ok(Self { bits })
    }

    pub fn quantize(&self, embedding: &Embedding) -> Result<QuantizedEmbedding> {
        let max_abs = embedding
            .values()
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let max_integer = ((1_i32 << (self.bits - 1)) - 1) as f32;
        let scale = if max_abs <= f32::EPSILON {
            1.0
        } else {
            max_abs / max_integer
        };
        let values = embedding
            .values()
            .iter()
            .map(|value| {
                let q = (*value / scale).round().clamp(-max_integer, max_integer);
                q as i8
            })
            .collect();
        Ok(QuantizedEmbedding { values, scale })
    }

    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Result<Embedding> {
        if quantized.values.is_empty() {
            return Err(SdkError::EmptyFeatures);
        }
        if quantized.values.len() > 8_192 {
            return Err(SdkError::DimensionLimit {
                actual: quantized.values.len(),
                max: 8_192,
            });
        }
        if !quantized.scale.is_finite() || quantized.scale <= 0.0 {
            return Err(SdkError::InvalidArgument(
                "quantization scale must be finite and positive".into(),
            ));
        }
        Embedding::new(
            quantized
                .values
                .iter()
                .map(|value| f32::from(*value) * quantized.scale)
                .collect(),
        )
    }

    pub fn fake_quantize(&self, embedding: &Embedding) -> Result<Embedding> {
        self.dequantize(&self.quantize(embedding)?)
    }
}
