use crate::{
    ConsistencyReport, DeterministicVerifier, Observation, OntologyGraph, Pipeline, Prediction,
    ReplayStatus, Result, ResultCertificate, SigintVerificationPolicy, VerificationContext,
    semantic_graph_for_prediction,
};

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedPrediction {
    prediction: Prediction,
    ontology: OntologyGraph,
    consistency: ConsistencyReport,
    certificate: ResultCertificate,
}

impl VerifiedPrediction {
    pub fn prediction(&self) -> &Prediction {
        &self.prediction
    }
    pub fn ontology(&self) -> &OntologyGraph {
        &self.ontology
    }
    pub fn consistency(&self) -> &ConsistencyReport {
        &self.consistency
    }
    pub fn certificate(&self) -> &ResultCertificate {
        &self.certificate
    }
}

#[derive(Clone)]
pub struct VerifiedPipeline {
    pipeline: Pipeline,
    context: VerificationContext,
    policy: SigintVerificationPolicy,
}

impl VerifiedPipeline {
    pub fn new(
        pipeline: Pipeline,
        context: VerificationContext,
        policy: SigintVerificationPolicy,
    ) -> Self {
        Self {
            pipeline,
            context,
            policy,
        }
    }

    pub fn predict(&self, observation: &Observation) -> Result<VerifiedPrediction> {
        let prediction = self.pipeline.predict(observation)?;
        let ontology = semantic_graph_for_prediction(observation, &prediction)?;
        let consistency = ontology.validate_reference_schema();
        let certificate = DeterministicVerifier::verify_sigint(
            observation,
            &prediction,
            &self.context,
            &self.policy,
            &consistency,
        )?;
        Ok(VerifiedPrediction {
            prediction,
            ontology,
            consistency,
            certificate,
        })
    }

    pub fn replay(
        &self,
        observation: &Observation,
        original: &ResultCertificate,
    ) -> Result<(VerifiedPrediction, ReplayStatus)> {
        let replayed = self.predict(observation)?;
        let status = DeterministicVerifier::compare_replay(original, replayed.certificate());
        Ok((replayed, status))
    }
}
