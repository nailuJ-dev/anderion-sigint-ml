use std::collections::BTreeMap;

use crate::{ClassScore, Classifier, Embedding, PrototypeClassifier, Result, SdkError};

#[derive(Debug, Clone)]
pub struct OnlinePrototypeClassifier {
    dimension: usize,
    prototypes: BTreeMap<String, Vec<f32>>,
    counts: BTreeMap<String, u64>,
    temperature: f32,
}

impl OnlinePrototypeClassifier {
    pub fn from_classifier(classifier: PrototypeClassifier) -> Self {
        Self {
            dimension: classifier.dimension(),
            prototypes: classifier.prototypes().clone(),
            counts: classifier.counts().clone(),
            temperature: classifier.temperature(),
        }
    }

    pub fn update(&mut self, label: impl Into<String>, embedding: &Embedding) -> Result<()> {
        if embedding.dim() != self.dimension {
            return Err(SdkError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.dim(),
            });
        }
        let label = label.into();
        if label.trim().is_empty() {
            return Err(SdkError::InvalidArgument(
                "class label must not be empty".into(),
            ));
        }
        let count = self.counts.entry(label.clone()).or_insert(0);
        let prototype = self
            .prototypes
            .entry(label)
            .or_insert_with(|| vec![0.0; self.dimension]);
        let next = count.saturating_add(1);
        let alpha = 1.0 / next as f32;
        for (dst, src) in prototype.iter_mut().zip(embedding.values()) {
            *dst += alpha * (*src - *dst);
        }
        *count = next;
        Ok(())
    }
}

impl Classifier for OnlinePrototypeClassifier {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>> {
        let classifier = PrototypeClassifier::fit(
            &self
                .prototypes
                .iter()
                .map(|(label, prototype)| (Embedding::new(prototype.clone()), label.clone()))
                .map(|(embedding, label)| embedding.map(|e| (e, label)))
                .collect::<Result<Vec<_>>>()?,
        )?
        .with_temperature(self.temperature)?;
        classifier.classify(embedding)
    }

    fn input_dim(&self) -> usize {
        self.dimension
    }
}
