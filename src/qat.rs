use crate::{ClassScore, Classifier, Embedding, PrototypeClassifier, Result, SymmetricQuantizer};

/// Reference classifier that applies fake symmetric quantization to both
/// training and inference embeddings before prototype learning/scoring.
#[derive(Debug, Clone)]
pub struct QuantizationAwarePrototypeClassifier {
    quantizer: SymmetricQuantizer,
    classifier: PrototypeClassifier,
}

impl QuantizationAwarePrototypeClassifier {
    pub fn fit(samples: &[(Embedding, String)], bits: u8) -> Result<Self> {
        let quantizer = SymmetricQuantizer::new(bits)?;
        let mut quantized_samples = Vec::with_capacity(samples.len());
        for (embedding, label) in samples {
            quantized_samples.push((quantizer.fake_quantize(embedding)?, label.clone()));
        }
        let classifier = PrototypeClassifier::fit(&quantized_samples)?;
        Ok(Self {
            quantizer,
            classifier,
        })
    }
}

impl Classifier for QuantizationAwarePrototypeClassifier {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>> {
        self.classifier
            .classify(&self.quantizer.fake_quantize(embedding)?)
    }

    fn input_dim(&self) -> usize {
        self.classifier.input_dim()
    }
}
