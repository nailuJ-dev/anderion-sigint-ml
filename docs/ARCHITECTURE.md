# Architecture

The SDK uses explicit validated data contracts and small ML components composed through public Rust traits. `Observation` validates caller features; `Encoder` creates `Embedding`; classifiers, anomaly/OOD models, calibration and uncertainty operate on embeddings; `Pipeline` composes inference without owning acquisition or transport.

P1 adds self-supervised masked-context learning, domain adaptation, temporal segmentation/sequence learning, metric learning, explainability and drift monitoring.

P2 adds four independent groups:

1. Representation: `FoundationPooler`, `TemporalSelfAttentionEncoder`, `TemporalTransformerEncoder`, `VarianceAutoencoder`, `ContrastiveProjector`.
2. Learning workflows: active-learning selection, embedding-prototype zero-shot classification, soft-label distillation and deterministic synthetic adapters.
3. Compression: symmetric quantization, fake-quantization simulation and magnitude pruning.
4. Deployment evaluation: edge parameter profiles, encoder benchmarks and wasm32 core validation.

All P2 reference algorithms operate on generic numeric vectors. No module loads a remote model, opens a network connection or requires an external runtime.

Security boundaries are enforced at constructors, artifact loaders and service request limits. Deserialization is followed by model-specific invariant validation before a model can enter the serving path.

## Reliability layer (0.3)

The opt-in reliability layer is split into four independent modules: `ontology` for typed semantic consistency, `pattern` for deterministic recurrence analysis, `verification` for canonical digests/certificates/replay, and `verified_pipeline` for composition with the existing inference pipeline. The existing ML pipeline remains unchanged.

