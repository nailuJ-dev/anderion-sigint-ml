## 0.6.0

- Fix reference I/Q spectral coverage across the full complex Nyquist interval.
- Add explicit `IqFeatureSchema` compatibility versioning; missing legacy schema metadata maps to v1, while new extractors default to v2.
- Add Hann-windowed complex FFT, fftshift and coarse full-band power aggregation.
- Add scientific regression tests for signed frequency separation, capture-length stability, normalization and amplitude invariance.
- Pin the repository toolchain to Rust 1.85.0 and correct the Golden Path binary command.
- Legacy v0.5 spectral artifacts require explicit v1 use or retraining/revalidation.

## 0.5.0

- Dense-spectrum component extraction.
- Receiver-aware RF normalization.
- Session-scoped open-world emitter discovery and controlled local enrollment.
- Deterministic local evidence graph.
- Compact spectrum encoder interface and reference backend.


## 0.4.0

- Add a one-command Golden Path from bounded raw I/Q captures to verified classification.
- Add deterministic reference I/Q feature extraction and bundled synthetic train/evaluation fixtures.
- Add Golden Path model/evaluation APIs, replay proof, CLI and CI smoke gate.
- Fixture metrics are explicitly non-operational and must not be reported as field accuracy.

# Changelog

## 0.4.1

- Add cross-repository integration CI against `spectra-sim`.
- Validate the simulator-to-SDK Golden Path contract on every main-branch change and pull request.


## 0.3.0 - 2026-08-11

- Added a typed, bounded micro-ontology and deterministic reference-schema consistency checks.
- Added deterministic recurring-sequence and co-occurrence pattern detection.
- Added canonical SHA-256 inference certificates, fixed-point decision digests and replay classification.
- Added opt-in verified pipeline wrappers without changing existing inference APIs.
- Kept the SIGINT SDK fully standalone with no private or remote runtime dependency.

## 0.2.0 - 2026-08-11

- Added standalone P2 foundation/attention/transformer, autoencoder and contrastive representation modules.
- Added active learning, zero-shot embedding classification and soft-label distillation.
- Added quantization, quantization-aware reference training, pruning, edge profiling/benchmarking and deterministic synthetic adapters.
- Added wasm32 core CI validation and clarified the standalone repository boundary.

## 0.1.0 - 2026-08-11

- Initial P0/P1 reference SDK.
