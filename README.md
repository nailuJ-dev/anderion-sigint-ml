# Anderion SIGINT ML

## Golden Path quickstart

Run an end-to-end raw-I/Q reference scenario with one command:

```bash
./scripts/run_golden_path.sh
```

This trains the bundled public reference model locally, classifies a held-out synthetic I/Q capture, verifies ontology consistency, performs an exact deterministic replay and writes `artifacts/golden-path/result.json`. See `docs/GOLDEN_PATH.md`.


A standalone, sensor-agnostic Rust SDK for machine-learning workflows over user-supplied signal feature vectors. The repository is self-contained: builds, tests, model artifacts, reference algorithms and inference APIs require no hidden service, non-public crate, external model registry or closed runtime component.

## Scope

The SDK starts from numerical feature vectors supplied by the caller. It does not acquire or intercept communications, decode content, exploit protocols, jam emitters, control radios or manipulate the electromagnetic environment. Its public surface is ML representation, classification, novelty/anomaly detection, temporal learning, adaptation, compression, evaluation and deployment support.

## P0

- validated `Observation`, `Embedding`, `ClassScore`, `Prediction` contracts;
- pluggable `Encoder`, `Classifier`, `AnomalyDetector`, `Calibrator` traits;
- deterministic reference encoder;
- supervised and few-shot prototype classification;
- confidence abstention, OOD/open-set recognition and anomaly scoring;
- calibration, uncertainty, similarity search and k-means;
- temporal aggregation, batch and streaming inference;
- CPU backend plus public external-backend contract;
- bounded SHA-256 verified artifacts and model cards;
- benchmark utilities and optional stateless HTTP service.

## P1

- masked-context self-supervised learning;
- continual prototype learning;
- mean/variance domain adaptation;
- diagonal metric learning;
- learned temporal change-point segmentation;
- sequence classification and weighted ensembles;
- feature-occlusion explainability;
- embedding drift monitoring;
- dataset manifests and leakage-resistant grouped splitting.

## P2

- `FoundationModel` contract and bounded `FoundationPooler` reference implementation;
- temporal self-attention and transformer-style residual encoder;
- variance-bottleneck autoencoder with reconstruction scoring;
- contrastive representation projector;
- uncertainty/diversity active-learning selection;
- zero-shot classification from caller-supplied embedding prototypes;
- soft-label knowledge distillation into a compact prototype student;
- symmetric 2–8 bit fake quantization / int8 representation;
- quantization-aware reference prototype training using fake-quantized embeddings;
- deterministic magnitude pruning and sparsity metrics;
- edge parameter profiling and encoder latency benchmarking;
- generic deterministic synthetic-data adapter;
- wasm32 core compile gate for non-server deployments.

## Quick start

```bash
cargo run --example basic
cargo test --all-features
```

Optional HTTP inference service:

```bash
cargo run --features server --example build_reference_bundle
SIGINT_MODEL_MANIFEST=artifacts/reference-model.manifest.json \
SIGINT_MODEL_PAYLOAD=artifacts/reference-model.json \
cargo run --features server --bin anderion-sigint-serve
```

The direct binary binds to loopback by default. Container deployments should place the service behind authenticated TLS ingress and apply request/body/resource limits described in `docs/OPERATIONS.md`.

## Standalone architecture

```text
Caller feature vector
        |
        v
   Encoder / representation learning
        |
        v
     Embedding
   /    |      \
  v     v       v
Class  OOD   Anomaly/Similarity
  \     |       /
   \    v      /
 Calibration + uncertainty
         |
         v
     Prediction

P2 side workflows:
foundation pooling | temporal attention | autoencoding | contrastive learning
active learning | distillation | quantization/QAT | pruning | edge profiling
```

Every component is implemented in this repository or supplied explicitly by the SDK caller through a public trait. There is no implicit network access, remote model resolution or hidden dependency path.

## Security

`unsafe` is forbidden. Production source denies `unwrap`, `expect` and `panic` through Clippy. Inputs, dimensions and artifacts are bounded; model payloads are checked for schema/type/length/SHA-256 and revalidated after deserialization. CI includes format, Clippy, tests, rustdoc, MSRV, dependency policy and a wasm32 core check.

See `SECURITY.md`, `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md` and `docs/P0_P1_P2_COVERAGE.md`.

## License

Apache-2.0.

## Ontology, recurring patterns, and deterministic verification

Version 0.3.0 adds an opt-in, self-contained reliability layer. `OntologyGraph` provides a small typed semantic schema for observations, signal events, classes, embeddings, evidence and recurring patterns. `PatternEngine` detects deterministic repeated sequences and co-occurrences with stable ordering. `VerifiedPipeline` wraps the existing `Pipeline` without changing its behavior and emits a `ResultCertificate` binding the canonical input, model/config digests, ontology/pipeline versions, exact result and deterministic decision digest.

`VerifiedPipeline::replay` reruns inference and returns `ReplayStatus::Exact`, `DecisionEquivalent`, or `NonReproducible`. Exact replay compares canonical SHA-256 result digests. Decision-equivalent replay uses fixed-point decision values and is intentionally weaker than exact replay. Neither mode proves that a model prediction is physically correct; they prove repeatability and policy consistency for the supplied model, input, configuration and runtime behavior.

The ontology and verifier require no external graph database, remote model service, private registry or non-public runtime.



## Physics simulator compatibility

This release is continuously tested against the public `spectra-sim` contract. Default compatibility target: `spectra-sim 0.1.1`. The integration is file-based JSON only; there is no Cargo or private-service dependency.

## Compatibility

| SDK | spectra-sim |
|---|---|
| 0.4.1 | 0.1.1 |
