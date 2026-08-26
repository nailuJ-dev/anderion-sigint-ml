use std::sync::Arc;

use crate::{ClassScore, Classifier, Embedding, Result, SdkError, TemporalVote};

#[derive(Clone)]
pub struct SequenceClassifier {
    classifier: Arc<dyn Classifier>,
    smoothing_window: usize,
}

impl SequenceClassifier {
    pub fn new(classifier: Arc<dyn Classifier>, smoothing_window: usize) -> Result<Self> {
        if smoothing_window == 0 || smoothing_window > 1024 {
            return Err(SdkError::InvalidArgument(
                "smoothing_window must be in 1..=1024".into(),
            ));
        }
        Ok(Self {
            classifier,
            smoothing_window,
        })
    }

    pub fn classify_sequence(&self, sequence: &[Embedding]) -> Result<Vec<Vec<ClassScore>>> {
        let vote = TemporalVote::new(0.85)?;
        let mut raw = Vec::with_capacity(sequence.len());
        let mut out = Vec::with_capacity(sequence.len());
        for embedding in sequence {
            raw.push(self.classifier.classify(embedding)?);
            let start = raw.len().saturating_sub(self.smoothing_window);
            out.push(vote.aggregate(&raw[start..])?);
        }
        Ok(out)
    }
}
