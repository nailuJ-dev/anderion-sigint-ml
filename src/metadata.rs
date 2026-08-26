use serde::{Deserialize, Serialize};

use crate::{Result, SdkError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    pub model_id: String,
    pub model_version: String,
    pub model_type: String,
    pub intended_use: String,
    pub input_dimension: usize,
    pub output_labels: Vec<String>,
    pub training_data_statement: String,
    pub limitations: Vec<String>,
    pub license: String,
}

impl ModelCard {
    pub fn validate(&self) -> Result<()> {
        if self.model_id.trim().is_empty()
            || self.model_version.trim().is_empty()
            || self.model_type.trim().is_empty()
        {
            return Err(SdkError::InvalidArgument(
                "model card identifiers must be non-empty".into(),
            ));
        }
        if self.input_dimension == 0 {
            return Err(SdkError::InvalidArgument(
                "model card input_dimension must be positive".into(),
            ));
        }
        if self.intended_use.trim().is_empty()
            || self.training_data_statement.trim().is_empty()
            || self.license.trim().is_empty()
        {
            return Err(SdkError::InvalidArgument(
                "model card narrative fields must be non-empty".into(),
            ));
        }
        Ok(())
    }
}
