use anderion_sigint_ml::{EvidenceEdgeKind, SessionEvidenceGraph};

#[test]
fn why_returns_support_and_contradiction_edges() {
    let mut graph = SessionEvidenceGraph::default();
    let decision = graph.add_node("decision", "emitter-A");
    let support = graph.add_node("observation", "capture-1");
    let contradiction = graph.add_node("observation", "capture-2");
    graph.add_edge(support, decision, EvidenceEdgeKind::Supports).unwrap();
    graph.add_edge(contradiction, decision, EvidenceEdgeKind::Contradicts).unwrap();
    let why = graph.why(decision).unwrap();
    assert_eq!(why.supporting.len(), 1);
    assert_eq!(why.contradicting.len(), 1);
}
