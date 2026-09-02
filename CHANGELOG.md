# Changelog

All notable changes to this project are documented here. Versions are listed in
reverse-chronological order.

## 0.7.0 - unreleased

### Security
- Re-run every constructor on deserialization (`#[serde(try_from = ...)]`) for
  `Observation`, `Embedding`, `ClassScore`, `Prediction`, `IqSample`,
  `IqCapture`, `ReferenceIqFeatureExtractor`, `DenseSpectrumExtractor`,
  `VerificationContext`, `SigintVerificationPolicy` and `ResultCertificate`.
  Bounds, finiteness and probability ranges were previously bypassed by any
  JSON path.
- `ResultCertificate` now carries a `self_digest` over all other fields,
  recomputed and checked on deserialization. Certificate schema version is
  bumped to 2; version 1 certificates are rejected.
- Audit the committed `Cargo.lock` instead of regenerating it in CI.
- Harden the optional HTTP service: bounded concurrency (503 on overload),
  per-request timeout (504), CPU work moved off the async runtime, generic
  5xx bodies, and `deny_unknown_fields` on the request type.

### Fixed
- `Prediction::top()` is now correct for deserialized predictions; class scores
  are re-sorted with a deterministic label tiebreak.
- Replay classification no longer downgrades an ontology-consistency divergence
  to `DecisionEquivalent`; it is a hard `NonReproducible`.
- `VerificationContext::config_digest` is derived from the actual reference
  configuration (extractor, schema, embedding dimension, thresholds, policy)
  instead of a frozen literal.
- The Golden Path verification policy uses a real uncertainty gate
  (`max_uncertainty = 0.95`); the previous value of `1.0` was unreachable.
- `DenseSpectrumExtractor` uses the same signed-frequency convention as the
  reference I/Q extractor, applies a Hann window before transforming, and no
  longer emits one component per plateau cell.
- `SignalComponent::bandwidth_hz` is a measured -3 dB width and
  `start_sample`/`end_sample` are a measured -3 dB time support; both were
  previously constant placeholders.
- `load_golden_sigint_training` validates by reference instead of deep-cloning
  every capture.
- `panic = "abort"` moved off the default release profile to a dedicated
  `release-edge` profile, so a handler panic can no longer kill the service.

### Changed
- **Breaking:** `IqCapture` requires at least `MIN_IQ_SAMPLES` (16) samples. A
  symmetric Hann window zeroes shorter captures instead of merely degrading
  them.
- **Breaking:** `ReferenceIqFeatureExtractor::extract` rejects
  `spectrum_bins > capture length`.
- **Breaking:** certificates issued by 0.6.x do not replay against 0.7.0
  (algorithm version and config digest both changed). Re-issue them.
- Docker images build with `--locked`, use BuildKit cache mounts, ship from a
  distroless runtime for both the service and the Golden Path demo, and honour
  a `.dockerignore`.
- `-D warnings` is enforced in CI only, not in `.cargo/config.toml`.

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

## 0.4.1

- Add cross-repository integration CI against `spectra-sim`.
- Validate the simulator-to-SDK Golden Path contract on every main-branch change and pull request.

## 0.4.0

- Add a one-command Golden Path from bounded raw I/Q captures to verified classification.
- Add deterministic reference I/Q feature extraction and bundled synthetic train/evaluation fixtures.
- Add Golden Path model/evaluation APIs, replay proof, CLI and CI smoke gate.
- Fixture metrics are explicitly non-operational and must not be reported as field accuracy.

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
