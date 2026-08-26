use crate::{ClassScore, Embedding, Observation, Result};

pub trait Encoder: Send + Sync {
    fn encode(&self, observation: &Observation) -> Result<Embedding>;
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
}

pub trait Classifier: Send + Sync {
    fn classify(&self, embedding: &Embedding) -> Result<Vec<ClassScore>>;
    fn input_dim(&self) -> usize;
}

pub trait AnomalyDetector: Send + Sync {
    fn anomaly_score(&self, embedding: &Embedding) -> Result<f32>;
    fn input_dim(&self) -> usize;
}

pub trait Calibrator: Send + Sync {
    fn calibrate(&self, scores: &[ClassScore]) -> Result<Vec<ClassScore>>;
}
