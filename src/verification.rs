use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};

use crate::{ConsistencyReport, Observation, Prediction, Result, SdkError};

const MAX_CONTEXT_TEXT_BYTES: usize = 256;
const MAX_FIXED_POINT_SCALE: u32 = 1_000_000_000;
const CERTIFICATE_ALGORITHM_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        finish_hash(hasher)
    }

    pub fn from_hex(value: &str) -> Result<Self> {
        let bytes = value.as_bytes();
        if bytes.len() != 64 {
            return Err(SdkError::InvalidArgument(
                "SHA-256 hex digest must contain 64 characters".into(),
            ));
        }
        let mut output = [0_u8; 32];
        for index in 0..32 {
            let high = hex_nibble(bytes[index * 2])?;
            let low = hex_nibble(bytes[index * 2 + 1])?;
            output[index] = (high << 4) | low;
        }
        Ok(Self(output))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        crate::io::hex_encode(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "VerificationContextWire")]
pub struct VerificationContext {
    model_digest: Digest32,
    config_digest: Digest32,
    ontology_version: String,
    pipeline_version: String,
    seed: u64,
}

impl VerificationContext {
    pub fn new(
        model_digest: Digest32,
        config_digest: Digest32,
        ontology_version: impl Into<String>,
        pipeline_version: impl Into<String>,
        seed: u64,
    ) -> Result<Self> {
        let ontology_version = ontology_version.into();
        let pipeline_version = pipeline_version.into();
        validate_context_text("ontology_version", &ontology_version)?;
        validate_context_text("pipeline_version", &pipeline_version)?;
        Ok(Self {
            model_digest,
            config_digest,
            ontology_version,
            pipeline_version,
            seed,
        })
    }

    pub fn model_digest(&self) -> Digest32 {
        self.model_digest
    }
    pub fn config_digest(&self) -> Digest32 {
        self.config_digest
    }
    pub fn ontology_version(&self) -> &str {
        &self.ontology_version
    }
    pub fn pipeline_version(&self) -> &str {
        &self.pipeline_version
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SigintVerificationPolicyWire")]
pub struct SigintVerificationPolicy {
    min_class_confidence: f32,
    max_uncertainty: f32,
    max_anomaly_score: Option<f32>,
    fixed_point_scale: u32,
    require_ontology_consistency: bool,
}

impl SigintVerificationPolicy {
    pub fn new(
        min_class_confidence: f32,
        max_uncertainty: f32,
        max_anomaly_score: Option<f32>,
        fixed_point_scale: u32,
        require_ontology_consistency: bool,
    ) -> Result<Self> {
        validate_probability(min_class_confidence)?;
        validate_probability(max_uncertainty)?;
        if let Some(value) = max_anomaly_score {
            validate_probability(value)?;
        }
        validate_scale(fixed_point_scale)?;
        Ok(Self {
            min_class_confidence,
            max_uncertainty,
            max_anomaly_score,
            fixed_point_scale,
            require_ontology_consistency,
        })
    }

    pub fn min_class_confidence(&self) -> f32 {
        self.min_class_confidence
    }
    pub fn max_uncertainty(&self) -> f32 {
        self.max_uncertainty
    }
    pub fn max_anomaly_score(&self) -> Option<f32> {
        self.max_anomaly_score
    }
    pub fn fixed_point_scale(&self) -> u32 {
        self.fixed_point_scale
    }
    pub fn requires_ontology_consistency(&self) -> bool {
        self.require_ontology_consistency
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationDecision {
    Accept,
    Abstain,
    Review,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayStatus {
    Exact,
    DecisionEquivalent,
    NonReproducible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ResultCertificateWire")]
pub struct ResultCertificate {
    algorithm_version: u32,
    input_digest: Digest32,
    context_digest: Digest32,
    policy_digest: Digest32,
    exact_result_digest: Digest32,
    decision_digest: Digest32,
    decision: VerificationDecision,
    ontology_valid: bool,
    ontology_violation_count: usize,
    /// SHA-256 over every other field. Recomputed and checked on deserialization
    /// so a certificate cannot be assembled field-by-field from untrusted JSON.
    /// This is an integrity check, not a signature: it proves the record was not
    /// hand-edited, not that it came from a trusted issuer.
    self_digest: Digest32,
}

impl ResultCertificate {
    pub fn algorithm_version(&self) -> u32 {
        self.algorithm_version
    }
    pub fn input_digest(&self) -> Digest32 {
        self.input_digest
    }
    pub fn context_digest(&self) -> Digest32 {
        self.context_digest
    }
    pub fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }
    pub fn exact_result_digest(&self) -> Digest32 {
        self.exact_result_digest
    }
    pub fn decision_digest(&self) -> Digest32 {
        self.decision_digest
    }
    pub fn decision(&self) -> VerificationDecision {
        self.decision
    }
    pub fn ontology_valid(&self) -> bool {
        self.ontology_valid
    }
    pub fn ontology_violation_count(&self) -> usize {
        self.ontology_violation_count
    }
    pub fn self_digest(&self) -> Digest32 {
        self.self_digest
    }

    #[allow(clippy::too_many_arguments)]
    fn seal(
        algorithm_version: u32,
        input_digest: Digest32,
        context_digest: Digest32,
        policy_digest: Digest32,
        exact_result_digest: Digest32,
        decision_digest: Digest32,
        decision: VerificationDecision,
        ontology_valid: bool,
        ontology_violation_count: usize,
    ) -> Self {
        let self_digest = hash_certificate_body(
            algorithm_version,
            input_digest,
            context_digest,
            policy_digest,
            exact_result_digest,
            decision_digest,
            decision,
            ontology_valid,
            ontology_violation_count,
        );
        Self {
            algorithm_version,
            input_digest,
            context_digest,
            policy_digest,
            exact_result_digest,
            decision_digest,
            decision,
            ontology_valid,
            ontology_violation_count,
            self_digest,
        }
    }
}

pub struct DeterministicVerifier;

impl DeterministicVerifier {
    pub fn verify_sigint(
        observation: &Observation,
        prediction: &Prediction,
        context: &VerificationContext,
        policy: &SigintVerificationPolicy,
        consistency: &ConsistencyReport,
    ) -> Result<ResultCertificate> {
        validate_context(context)?;
        validate_policy(policy)?;
        validate_prediction(prediction)?;
        let top = prediction
            .top()
            .ok_or_else(|| SdkError::InvalidArgument("prediction has no top class".into()))?;
        let decision = if policy.require_ontology_consistency && !consistency.is_valid() {
            VerificationDecision::Review
        } else if prediction.is_unknown()
            || prediction.uncertainty() > policy.max_uncertainty
            || top.probability() < policy.min_class_confidence
        {
            VerificationDecision::Abstain
        } else if policy.max_anomaly_score.is_some_and(|limit| {
            prediction
                .anomaly_score()
                .is_some_and(|score| score > limit)
        }) {
            VerificationDecision::Review
        } else {
            VerificationDecision::Accept
        };
        Ok(ResultCertificate::seal(
            CERTIFICATE_ALGORITHM_VERSION,
            hash_observation(observation),
            hash_context(context),
            hash_policy(policy),
            hash_prediction_exact(prediction),
            hash_prediction_decision(prediction, policy.fixed_point_scale)?,
            decision,
            consistency.is_valid(),
            consistency.violations().len(),
        ))
    }

    pub fn compare_replay(
        original: &ResultCertificate,
        replayed: &ResultCertificate,
    ) -> ReplayStatus {
        // Any divergence in the bound context, the applied policy, the taken
        // decision, or the semantic-consistency outcome is a hard failure. Only a
        // difference in the exact numeric result may be downgraded to
        // `DecisionEquivalent`.
        if original.algorithm_version != replayed.algorithm_version
            || original.input_digest != replayed.input_digest
            || original.context_digest != replayed.context_digest
            || original.policy_digest != replayed.policy_digest
            || original.decision != replayed.decision
            || original.ontology_valid != replayed.ontology_valid
            || original.ontology_violation_count != replayed.ontology_violation_count
        {
            return ReplayStatus::NonReproducible;
        }
        if original.exact_result_digest == replayed.exact_result_digest {
            ReplayStatus::Exact
        } else if original.decision_digest == replayed.decision_digest {
            ReplayStatus::DecisionEquivalent
        } else {
            ReplayStatus::NonReproducible
        }
    }
}

fn validate_prediction(prediction: &Prediction) -> Result<()> {
    if prediction.scores().is_empty() {
        return Err(SdkError::InvalidArgument(
            "prediction must contain class scores".into(),
        ));
    }
    for score in prediction.scores() {
        if score.label().trim().is_empty() || score.label().len() > 4_096 {
            return Err(SdkError::InvalidArgument(
                "prediction class label is invalid".into(),
            ));
        }
        validate_probability(score.probability())?;
    }
    validate_probability(prediction.uncertainty())?;
    if let Some(value) = prediction.anomaly_score() {
        validate_probability(value)?;
    }
    if prediction.embedding().values().is_empty() || prediction.embedding().values().len() > 8_192 {
        return Err(SdkError::DimensionLimit {
            actual: prediction.embedding().values().len(),
            max: 8_192,
        });
    }
    if let Some((index, _)) = prediction
        .embedding()
        .values()
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SdkError::NonFiniteValue { index });
    }
    Ok(())
}

fn validate_context(context: &VerificationContext) -> Result<()> {
    validate_context_text("ontology_version", context.ontology_version())?;
    validate_context_text("pipeline_version", context.pipeline_version())?;
    Ok(())
}

fn validate_policy(policy: &SigintVerificationPolicy) -> Result<()> {
    validate_probability(policy.min_class_confidence)?;
    validate_probability(policy.max_uncertainty)?;
    if let Some(value) = policy.max_anomaly_score {
        validate_probability(value)?;
    }
    validate_scale(policy.fixed_point_scale)
}

fn hash_policy(policy: &SigintVerificationPolicy) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-sigint-verification-policy-v1");
    update_f32(&mut hasher, policy.min_class_confidence);
    update_f32(&mut hasher, policy.max_uncertainty);
    match policy.max_anomaly_score {
        Some(value) => {
            hasher.update([1_u8]);
            update_f32(&mut hasher, value);
        }
        None => hasher.update([0_u8]),
    }
    hasher.update(policy.fixed_point_scale.to_le_bytes());
    hasher.update([u8::from(policy.require_ontology_consistency)]);
    finish_hash(hasher)
}

fn hash_observation(observation: &Observation) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-sigint-input-v1");
    update_bytes(&mut hasher, observation.id().as_bytes());
    hasher.update(observation.timestamp_ms().to_le_bytes());
    update_f32_slice(&mut hasher, observation.features());
    finish_hash(hasher)
}

fn hash_context(context: &VerificationContext) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-sigint-context-v1");
    hasher.update(context.model_digest.as_bytes());
    hasher.update(context.config_digest.as_bytes());
    update_bytes(&mut hasher, context.ontology_version.as_bytes());
    update_bytes(&mut hasher, context.pipeline_version.as_bytes());
    hasher.update(context.seed.to_le_bytes());
    finish_hash(hasher)
}

fn hash_prediction_exact(prediction: &Prediction) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-sigint-result-exact-v1");
    hasher.update((prediction.scores().len() as u64).to_le_bytes());
    for score in prediction.scores() {
        update_bytes(&mut hasher, score.label().as_bytes());
        update_f32(&mut hasher, score.probability());
    }
    hasher.update([u8::from(prediction.is_unknown())]);
    match prediction.anomaly_score() {
        Some(value) => {
            hasher.update([1_u8]);
            update_f32(&mut hasher, value);
        }
        None => hasher.update([0_u8]),
    }
    update_f32_slice(&mut hasher, prediction.embedding().values());
    update_f32(&mut hasher, prediction.uncertainty());
    finish_hash(hasher)
}

fn hash_prediction_decision(prediction: &Prediction, scale: u32) -> Result<Digest32> {
    let top = prediction
        .top()
        .ok_or_else(|| SdkError::InvalidArgument("prediction has no top class".into()))?;
    let mut hasher = domain_hasher(b"anderion-sigint-decision-v1");
    update_bytes(&mut hasher, top.label().as_bytes());
    hasher.update(quantize_probability(top.probability(), scale)?.to_le_bytes());
    hasher.update([u8::from(prediction.is_unknown())]);
    match prediction.anomaly_score() {
        Some(value) => {
            hasher.update([1_u8]);
            hasher.update(quantize_probability(value, scale)?.to_le_bytes());
        }
        None => hasher.update([0_u8]),
    }
    hasher.update(quantize_probability(prediction.uncertainty(), scale)?.to_le_bytes());
    Ok(finish_hash(hasher))
}

fn domain_hasher(domain: &[u8]) -> Sha256 {
    let mut hasher = Sha256::new();
    update_bytes(&mut hasher, domain);
    hasher
}

fn update_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn update_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        update_f32(hasher, *value);
    }
}

fn update_f32(hasher: &mut Sha256, value: f32) {
    let bits = if value == 0.0 { 0_u32 } else { value.to_bits() };
    hasher.update(bits.to_le_bytes());
}

fn finish_hash(hasher: Sha256) -> Digest32 {
    let finalized = hasher.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&finalized);
    Digest32(bytes)
}

fn quantize_probability(value: f32, scale: u32) -> Result<u64> {
    validate_probability(value)?;
    Ok(((value as f64) * (scale as f64)).round() as u64)
}

fn validate_probability(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(SdkError::InvalidProbability(value));
    }
    Ok(())
}

fn validate_scale(scale: u32) -> Result<()> {
    if scale == 0 || scale > MAX_FIXED_POINT_SCALE {
        return Err(SdkError::InvalidArgument(
            "fixed_point_scale must be in 1..=1_000_000_000".into(),
        ));
    }
    Ok(())
}

fn validate_context_text(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(SdkError::InvalidArgument(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > MAX_CONTEXT_TEXT_BYTES {
        return Err(SdkError::DimensionLimit {
            actual: value.len(),
            max: MAX_CONTEXT_TEXT_BYTES,
        });
    }
    Ok(())
}

fn hex_nibble(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(SdkError::InvalidArgument(
            "invalid SHA-256 hex digest".into(),
        )),
    }
}

fn decision_tag(decision: VerificationDecision) -> u8 {
    match decision {
        VerificationDecision::Accept => 0,
        VerificationDecision::Abstain => 1,
        VerificationDecision::Review => 2,
        VerificationDecision::Reject => 3,
    }
}

#[allow(clippy::too_many_arguments)]
fn hash_certificate_body(
    algorithm_version: u32,
    input_digest: Digest32,
    context_digest: Digest32,
    policy_digest: Digest32,
    exact_result_digest: Digest32,
    decision_digest: Digest32,
    decision: VerificationDecision,
    ontology_valid: bool,
    ontology_violation_count: usize,
) -> Digest32 {
    let mut hasher = domain_hasher(b"anderion-sigint-certificate-v2");
    hasher.update(algorithm_version.to_le_bytes());
    hasher.update(input_digest.as_bytes());
    hasher.update(context_digest.as_bytes());
    hasher.update(policy_digest.as_bytes());
    hasher.update(exact_result_digest.as_bytes());
    hasher.update(decision_digest.as_bytes());
    hasher.update([decision_tag(decision)]);
    hasher.update([u8::from(ontology_valid)]);
    hasher.update((ontology_violation_count as u64).to_le_bytes());
    finish_hash(hasher)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VerificationContextWire {
    model_digest: Digest32,
    config_digest: Digest32,
    ontology_version: String,
    pipeline_version: String,
    seed: u64,
}

impl TryFrom<VerificationContextWire> for VerificationContext {
    type Error = SdkError;

    fn try_from(wire: VerificationContextWire) -> Result<Self> {
        Self::new(
            wire.model_digest,
            wire.config_digest,
            wire.ontology_version,
            wire.pipeline_version,
            wire.seed,
        )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SigintVerificationPolicyWire {
    min_class_confidence: f32,
    max_uncertainty: f32,
    max_anomaly_score: Option<f32>,
    fixed_point_scale: u32,
    require_ontology_consistency: bool,
}

impl TryFrom<SigintVerificationPolicyWire> for SigintVerificationPolicy {
    type Error = SdkError;

    fn try_from(wire: SigintVerificationPolicyWire) -> Result<Self> {
        Self::new(
            wire.min_class_confidence,
            wire.max_uncertainty,
            wire.max_anomaly_score,
            wire.fixed_point_scale,
            wire.require_ontology_consistency,
        )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultCertificateWire {
    algorithm_version: u32,
    input_digest: Digest32,
    context_digest: Digest32,
    policy_digest: Digest32,
    exact_result_digest: Digest32,
    decision_digest: Digest32,
    decision: VerificationDecision,
    ontology_valid: bool,
    ontology_violation_count: usize,
    self_digest: Digest32,
}

impl TryFrom<ResultCertificateWire> for ResultCertificate {
    type Error = SdkError;

    fn try_from(wire: ResultCertificateWire) -> Result<Self> {
        if wire.algorithm_version != CERTIFICATE_ALGORITHM_VERSION {
            return Err(SdkError::SchemaMismatch {
                expected: CERTIFICATE_ALGORITHM_VERSION,
                actual: wire.algorithm_version,
            });
        }
        let sealed = Self::seal(
            wire.algorithm_version,
            wire.input_digest,
            wire.context_digest,
            wire.policy_digest,
            wire.exact_result_digest,
            wire.decision_digest,
            wire.decision,
            wire.ontology_valid,
            wire.ontology_violation_count,
        );
        if sealed.self_digest != wire.self_digest {
            return Err(SdkError::DigestMismatch);
        }
        Ok(sealed)
    }
}
