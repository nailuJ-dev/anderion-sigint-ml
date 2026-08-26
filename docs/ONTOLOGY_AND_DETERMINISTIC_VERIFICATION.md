# Ontology and Deterministic Verification

## Purpose

This layer adds semantic consistency, recurring-pattern discovery and reproducible verification to the standalone SIGINT ML SDK. It is opt-in and does not modify the existing `Pipeline` API.

## Lightweight ontology

The ontology is an in-process typed graph. It uses Rust enums and bounded `BTreeMap`/`Vec` storage rather than RDF/OWL, a graph server or a remote schema registry.

The reference schema contains these concepts: `Observation`, `FeatureSet`, `SignalEvent`, `SignalClass`, `Embedding`, `Pattern`, and `Evidence`.

Reference relations include observation containment, event classification, event embedding, pattern membership/composition and evidence support. `validate_reference_schema()` checks relation signatures and cardinality constraints with deterministic violation ordering. The graph is bounded to 65,536 nodes and 262,144 relations.

`semantic_graph_for_prediction()` converts a normal SDK prediction into the reference semantic representation. Callers can add their own pattern/evidence nodes and relations while retaining deterministic validation.

## Pattern engine

`PatternEngine` works on `PatternEvent` values. A `PatternToken` binds an ontology concept to a symbolic value and can optionally carry an embedding-cluster identifier. This lets callers combine semantic recurrence and ML embedding recurrence without coupling the SDK to a specific clustering implementation.

Two deterministic analyses are provided:

- repeated consecutive sequences, with configurable sequence length and minimum occurrence count;
- co-occurrence within fixed time buckets.

Input is sorted by timestamp and token before analysis. Results are ranked deterministically by occurrence count and canonical token ordering, so input iteration order does not change the result.

## Deterministic verification layer

`VerifiedPipeline` wraps an existing `Pipeline`. For every prediction it:

1. runs normal inference;
2. creates the semantic graph;
3. validates ontology consistency;
4. revalidates critical prediction invariants;
5. hashes the canonical input;
6. hashes the verification context containing model/config digests, ontology version, pipeline version and seed;
7. hashes the exact result using normalized binary representations;
8. creates a fixed-point decision digest;
9. applies the deterministic policy;
10. returns a `ResultCertificate`.

Policy decisions are `Accept`, `Abstain`, `Review`, or `Reject`. The reference SIGINT policy abstains for unknown/high-uncertainty/low-confidence results and requests review for semantic contradictions or excessive anomaly score.

## Replay

`VerifiedPipeline::replay()` reruns inference and compares the new certificate with a prior one:

- `Exact`: context, input, exact result, decision and ontology state match;
- `DecisionEquivalent`: context/input/decision match and the fixed-point decision digest matches, but the exact result digest differs;
- `NonReproducible`: context, input, decision or canonical result evidence is incompatible.

The exact digest uses IEEE representations with positive and negative zero canonicalized to the same zero value. The decision digest uses a caller-configured fixed-point scale and is deliberately weaker than exact replay.

## Security and guarantees

The verifier revalidates critical fields before issuing a certificate because deserialization can bypass normal constructors. Context strings, ontology graphs, pattern inputs and fixed-point scale are bounded.

A certificate proves repeatability of the canonical input/result and deterministic policy evaluation for the supplied model/configuration context. It does **not** prove that a model prediction is physically correct, that a dataset is unbiased, or that an arbitrary external inference backend is itself deterministic. When an external backend is nondeterministic, replay detects divergence instead of hiding it.

No graph database, private crate, private registry, remote inference service, hidden model format, or automatic network egress is required.
