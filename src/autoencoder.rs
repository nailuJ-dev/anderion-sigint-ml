use serde::{Deserialize, Serialize};

use crate::{Embedding, Result, SdkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceAutoencoder {
    input_dim: usize,
    bottleneck_dim: usize,
    mean: Vec<f32>,
    selected: Vec<usize>,
}

impl VarianceAutoencoder {
    pub fn fit(samples: &[Embedding], bottleneck_dim: usize) -> Result<Self> {
        if samples.is_empty() {
            return Err(SdkError::EmptyDataset);
        }
        let input_dim = samples[0].dim();
        if bottleneck_dim == 0 || bottleneck_dim > input_dim {
            return Err(SdkError::InvalidArgument(
                "bottleneck_dim must be in 1..=input_dim".into(),
            ));
        }
        if samples.iter().any(|sample| sample.dim() != input_dim) {
            let actual = samples
                .iter()
                .map(Embedding::dim)
                .max()
                .unwrap_or(input_dim);
            return Err(SdkError::DimensionMismatch {
                expected: input_dim,
                actual,
            });
        }
        let count = samples.len() as f32;
        let mut mean = vec![0.0_f32; input_dim];
        for sample in samples {
            for (dst, value) in mean.iter_mut().zip(sample.values()) {
                *dst += *value;
            }
        }
        for value in &mut mean {
            *value /= count;
        }
        let mut variances = vec![0.0_f32; input_dim];
        for sample in samples {
            for ((variance, value), center) in variances.iter_mut().zip(sample.values()).zip(&mean)
            {
                let delta = *value - *center;
                *variance += delta * delta;
            }
        }
        let mut ranked: Vec<(usize, f32)> = variances.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let selected = ranked
            .into_iter()
            .take(bottleneck_dim)
            .map(|(index, _)| index)
            .collect();
        let model = Self {
            input_dim,
            bottleneck_dim,
            mean,
            selected,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<()> {
        if self.input_dim == 0
            || self.input_dim > 8_192
            || self.bottleneck_dim == 0
            || self.bottleneck_dim > self.input_dim
        {
            return Err(SdkError::InvalidArgument(
                "invalid autoencoder dimensions".into(),
            ));
        }
        if self.mean.len() != self.input_dim || self.mean.iter().any(|value| !value.is_finite()) {
            return Err(SdkError::InvalidArgument(
                "invalid autoencoder mean vector".into(),
            ));
        }
        if self.selected.len() != self.bottleneck_dim
            || self.selected.iter().any(|index| *index >= self.input_dim)
        {
            return Err(SdkError::InvalidArgument(
                "invalid autoencoder selected indices".into(),
            ));
        }
        let mut sorted = self.selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != self.selected.len() {
            return Err(SdkError::InvalidArgument(
                "autoencoder selected indices must be unique".into(),
            ));
        }
        Ok(())
    }

    pub fn encode(&self, embedding: &Embedding) -> Result<Embedding> {
        self.validate()?;
        if embedding.dim() != self.input_dim {
            return Err(SdkError::DimensionMismatch {
                expected: self.input_dim,
                actual: embedding.dim(),
            });
        }
        let latent = self
            .selected
            .iter()
            .map(|index| embedding.values()[*index] - self.mean[*index])
            .collect();
        Embedding::new(latent)
    }

    pub fn decode(&self, latent: &Embedding) -> Result<Embedding> {
        self.validate()?;
        if latent.dim() != self.bottleneck_dim {
            return Err(SdkError::DimensionMismatch {
                expected: self.bottleneck_dim,
                actual: latent.dim(),
            });
        }
        let mut reconstructed = self.mean.clone();
        for (latent_value, index) in latent.values().iter().zip(&self.selected) {
            reconstructed[*index] += *latent_value;
        }
        Embedding::new(reconstructed)
    }

    pub fn reconstruction_error(&self, embedding: &Embedding) -> Result<f32> {
        let latent = self.encode(embedding)?;
        let reconstructed = self.decode(&latent)?;
        let mse = embedding
            .values()
            .iter()
            .zip(reconstructed.values())
            .map(|(a, b)| {
                let delta = a - b;
                delta * delta
            })
            .sum::<f32>()
            / self.input_dim as f32;
        Ok(mse)
    }
}
