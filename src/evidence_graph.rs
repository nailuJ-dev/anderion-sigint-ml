use crate::{Result, SdkError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceNode {
    pub id: u64,
    pub kind: String,
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceEdgeKind {
    Supports,
    Contradicts,
    DerivedFrom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceEdge {
    pub from: u64,
    pub to: u64,
    pub kind: EvidenceEdgeKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SessionEvidenceGraph {
    next_id: u64,
    nodes: Vec<EvidenceNode>,
    edges: Vec<EvidenceEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhyResult {
    pub decision: EvidenceNode,
    pub supporting: Vec<EvidenceNode>,
    pub contradicting: Vec<EvidenceNode>,
}

impl SessionEvidenceGraph {
    pub fn add_node(&mut self, kind: impl Into<String>, label: impl Into<String>) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.nodes.push(EvidenceNode {
            id,
            kind: kind.into(),
            label: label.into(),
        });
        id
    }

    pub fn add_edge(&mut self, from: u64, to: u64, kind: EvidenceEdgeKind) -> Result<()> {
        if !self.nodes.iter().any(|node| node.id == from)
            || !self.nodes.iter().any(|node| node.id == to)
        {
            return Err(SdkError::InvalidArgument(
                "evidence edge references an unknown node".into(),
            ));
        }
        self.edges.push(EvidenceEdge { from, to, kind });
        Ok(())
    }

    pub fn why(&self, decision_id: u64) -> Result<WhyResult> {
        let decision = self
            .nodes
            .iter()
            .find(|node| node.id == decision_id)
            .cloned()
            .ok_or_else(|| SdkError::InvalidArgument("decision node not found".into()))?;
        let mut supporting = Vec::new();
        let mut contradicting = Vec::new();
        for edge in self.edges.iter().filter(|edge| edge.to == decision_id) {
            let Some(node) = self.nodes.iter().find(|node| node.id == edge.from).cloned() else {
                continue;
            };
            match edge.kind {
                EvidenceEdgeKind::Supports => supporting.push(node),
                EvidenceEdgeKind::Contradicts => contradicting.push(node),
                EvidenceEdgeKind::DerivedFrom => {}
            }
        }
        Ok(WhyResult {
            decision,
            supporting,
            contradicting,
        })
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.next_id = 0;
    }
    pub fn nodes(&self) -> &[EvidenceNode] {
        &self.nodes
    }
    pub fn edges(&self) -> &[EvidenceEdge] {
        &self.edges
    }
}
