use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Observation, Prediction, Result, SdkError};

const MAX_NODES: usize = 65_536;
const MAX_RELATIONS: usize = 262_144;
const MAX_ID_BYTES: usize = 256;
const MAX_LABEL_BYTES: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConceptKind {
    Observation,
    FeatureSet,
    SignalEvent,
    SignalClass,
    Embedding,
    Pattern,
    Evidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RelationKind {
    Contains,
    ClassifiedAs,
    HasEmbedding,
    BelongsTo,
    ComposedOf,
    SupportedBy,
    Supports,
    DerivedFrom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OntologyNode {
    id: String,
    kind: ConceptKind,
    label: Option<String>,
}

impl OntologyNode {
    pub fn new(id: impl Into<String>, kind: ConceptKind, label: Option<String>) -> Result<Self> {
        let id = id.into();
        validate_text("ontology node id", &id, MAX_ID_BYTES)?;
        if let Some(value) = &label {
            validate_text("ontology node label", value, MAX_LABEL_BYTES)?;
        }
        Ok(Self { id, kind, label })
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn kind(&self) -> ConceptKind {
        self.kind
    }
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OntologyRelation {
    source: String,
    relation: RelationKind,
    target: String,
}

impl OntologyRelation {
    pub fn new(
        source: impl Into<String>,
        relation: RelationKind,
        target: impl Into<String>,
    ) -> Result<Self> {
        let source = source.into();
        let target = target.into();
        validate_text("ontology relation source", &source, MAX_ID_BYTES)?;
        validate_text("ontology relation target", &target, MAX_ID_BYTES)?;
        Ok(Self {
            source,
            relation,
            target,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn relation(&self) -> RelationKind {
        self.relation
    }
    pub fn target(&self) -> &str {
        &self.target
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConsistencyViolation {
    code: String,
    message: String,
}

impl ConsistencyViolation {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Result<Self> {
        let code = code.into();
        let message = message.into();
        validate_text("consistency code", &code, 128)?;
        validate_text("consistency message", &message, 2_048)?;
        Ok(Self { code, message })
    }

    pub fn code(&self) -> &str {
        &self.code
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsistencyReport {
    violations: Vec<ConsistencyViolation>,
}

impl ConsistencyReport {
    pub fn valid() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    pub fn from_violation(code: impl Into<String>, message: impl Into<String>) -> Result<Self> {
        Ok(Self {
            violations: vec![ConsistencyViolation::new(code, message)?],
        })
    }

    fn from_violations(mut violations: Vec<ConsistencyViolation>) -> Self {
        violations.sort();
        violations.dedup();
        Self { violations }
    }

    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }
    pub fn violations(&self) -> &[ConsistencyViolation] {
        &self.violations
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OntologyGraph {
    schema_version: String,
    nodes: BTreeMap<String, OntologyNode>,
    relations: Vec<OntologyRelation>,
}

impl OntologyGraph {
    pub fn new(schema_version: impl Into<String>) -> Result<Self> {
        let schema_version = schema_version.into();
        validate_text("ontology schema version", &schema_version, 128)?;
        Ok(Self {
            schema_version,
            nodes: BTreeMap::new(),
            relations: Vec::new(),
        })
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }
    pub fn nodes(&self) -> &BTreeMap<String, OntologyNode> {
        &self.nodes
    }
    pub fn relations(&self) -> &[OntologyRelation] {
        &self.relations
    }

    pub fn add_node(&mut self, node: OntologyNode) -> Result<()> {
        if self.nodes.len() >= MAX_NODES {
            return Err(SdkError::DimensionLimit {
                actual: self.nodes.len().saturating_add(1),
                max: MAX_NODES,
            });
        }
        if self.nodes.contains_key(node.id()) {
            return Err(SdkError::InvalidArgument(
                "duplicate ontology node id".into(),
            ));
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    pub fn add_relation(&mut self, relation: OntologyRelation) -> Result<()> {
        if self.relations.len() >= MAX_RELATIONS {
            return Err(SdkError::DimensionLimit {
                actual: self.relations.len().saturating_add(1),
                max: MAX_RELATIONS,
            });
        }
        if !self.nodes.contains_key(relation.source())
            || !self.nodes.contains_key(relation.target())
        {
            return Err(SdkError::InvalidArgument(
                "ontology relation endpoint does not exist".into(),
            ));
        }
        self.relations.push(relation);
        Ok(())
    }

    pub fn validate_reference_schema(&self) -> ConsistencyReport {
        let mut violations = Vec::new();
        if self.schema_version != "sigint-ontology-v1" {
            violations.push(raw_violation(
                "schema-version",
                "unexpected ontology schema version",
            ));
        }
        if self.nodes.len() > MAX_NODES {
            violations.push(raw_violation("node-limit", "ontology node limit exceeded"));
            return ConsistencyReport::from_violations(violations);
        }
        if self.relations.len() > MAX_RELATIONS {
            violations.push(raw_violation(
                "relation-limit",
                "ontology relation limit exceeded",
            ));
            return ConsistencyReport::from_violations(violations);
        }
        for (key, node) in &self.nodes {
            if key != node.id()
                || validate_text("ontology node id", node.id(), MAX_ID_BYTES).is_err()
            {
                violations.push(raw_violation(
                    "node-id",
                    "ontology node id is invalid or does not match its map key",
                ));
            }
            if node.label().is_some_and(|label| {
                validate_text("ontology node label", label, MAX_LABEL_BYTES).is_err()
            }) {
                violations.push(raw_violation(
                    "node-label",
                    "ontology node label is invalid",
                ));
            }
        }
        let mut cardinality: BTreeMap<(String, RelationKind), usize> = BTreeMap::new();

        for relation in &self.relations {
            let source = self.nodes.get(relation.source());
            let target = self.nodes.get(relation.target());
            let (Some(source), Some(target)) = (source, target) else {
                violations.push(raw_violation(
                    "missing-endpoint",
                    "ontology relation endpoint is missing",
                ));
                continue;
            };
            if !allowed_relation(source.kind(), relation.relation(), target.kind()) {
                violations.push(raw_violation(
                    "relation-schema",
                    &format!(
                        "relation {:?} is not allowed from {:?} to {:?}",
                        relation.relation(),
                        source.kind(),
                        target.kind()
                    ),
                ));
            }
            *cardinality
                .entry((relation.source.clone(), relation.relation()))
                .or_insert(0) += 1;
        }

        for ((source, relation), count) in cardinality {
            if max_cardinality(relation).is_some_and(|max| count > max) {
                violations.push(raw_violation(
                    "cardinality",
                    &format!("source {source} has {count} {:?} relations", relation),
                ));
            }
        }

        ConsistencyReport::from_violations(violations)
    }
}

pub fn semantic_graph_for_prediction(
    observation: &Observation,
    prediction: &Prediction,
) -> Result<OntologyGraph> {
    let top = prediction
        .top()
        .ok_or_else(|| SdkError::InvalidArgument("prediction has no top class".into()))?;
    let mut graph = OntologyGraph::new("sigint-ontology-v1")?;
    graph.add_node(OntologyNode::new(
        "observation:primary",
        ConceptKind::Observation,
        None,
    )?)?;
    graph.add_node(OntologyNode::new(
        "features:primary",
        ConceptKind::FeatureSet,
        None,
    )?)?;
    graph.add_node(OntologyNode::new(
        "event:primary",
        ConceptKind::SignalEvent,
        Some(observation.id().to_string()),
    )?)?;
    graph.add_node(OntologyNode::new(
        "class:primary",
        ConceptKind::SignalClass,
        Some(top.label().to_string()),
    )?)?;
    graph.add_node(OntologyNode::new(
        "embedding:primary",
        ConceptKind::Embedding,
        None,
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "observation:primary",
        RelationKind::Contains,
        "features:primary",
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "event:primary",
        RelationKind::DerivedFrom,
        "observation:primary",
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "event:primary",
        RelationKind::ClassifiedAs,
        "class:primary",
    )?)?;
    graph.add_relation(OntologyRelation::new(
        "event:primary",
        RelationKind::HasEmbedding,
        "embedding:primary",
    )?)?;
    Ok(graph)
}

fn allowed_relation(source: ConceptKind, relation: RelationKind, target: ConceptKind) -> bool {
    matches!(
        (source, relation, target),
        (
            ConceptKind::Observation,
            RelationKind::Contains,
            ConceptKind::FeatureSet
        ) | (
            ConceptKind::SignalEvent,
            RelationKind::DerivedFrom,
            ConceptKind::Observation
        ) | (
            ConceptKind::SignalEvent,
            RelationKind::ClassifiedAs,
            ConceptKind::SignalClass
        ) | (
            ConceptKind::SignalEvent,
            RelationKind::HasEmbedding,
            ConceptKind::Embedding
        ) | (
            ConceptKind::SignalEvent,
            RelationKind::BelongsTo,
            ConceptKind::Pattern
        ) | (
            ConceptKind::Pattern,
            RelationKind::ComposedOf,
            ConceptKind::SignalEvent
        ) | (
            ConceptKind::SignalEvent,
            RelationKind::SupportedBy,
            ConceptKind::Evidence
        ) | (
            ConceptKind::Evidence,
            RelationKind::Supports,
            ConceptKind::SignalEvent
        )
    )
}

fn max_cardinality(relation: RelationKind) -> Option<usize> {
    match relation {
        RelationKind::DerivedFrom | RelationKind::ClassifiedAs | RelationKind::HasEmbedding => {
            Some(1)
        }
        _ => None,
    }
}

fn raw_violation(code: &str, message: &str) -> ConsistencyViolation {
    ConsistencyViolation {
        code: code.to_string(),
        message: message.to_string(),
    }
}

fn validate_text(field: &str, value: &str, max_bytes: usize) -> Result<()> {
    if value.trim().is_empty() {
        return Err(SdkError::InvalidArgument(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > max_bytes {
        return Err(SdkError::DimensionLimit {
            actual: value.len(),
            max: max_bytes,
        });
    }
    Ok(())
}
