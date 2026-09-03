# anderion-sigint-ml

**Open-source machine-learning and RF-analysis primitives for understanding complex radio environments.**

`anderion-sigint-ml` is a Rust SDK for RF signal analysis, classification, anomaly detection and emitter-level reasoning. It operates on I/Q captures and feature vectors supplied by the caller.

The project targets a specific difficulty:

> RF systems rarely operate in clean, closed-set environments.

Signals overlap. Receivers change. Propagation distorts fingerprints. Unknown emitters appear. Models become overconfident. This SDK provides building blocks for systems that have to handle those conditions explicitly rather than hiding them behind a single argmax.

## Scope

This is a **signal-analysis library**. It is important to be precise about what that excludes.

The SDK does **not**:

* intercept, receive or transmit anything — it has no radio, driver or SDR integration
* control, tune, or configure any hardware
* demodulate or decode communications content
* perform jamming, spoofing, or any form of electronic attack
* geolocate emitters
* connect to any network at runtime, upload data, or emit telemetry
* maintain a persistent, cross-session or cross-site emitter identity database

It consumes I/Q samples or numeric feature vectors that the calling application already possesses, and returns classifications, embeddings, hypotheses and verification evidence. Everything upstream of the samples, and every operational use of the output, is the integrator's responsibility — including the legal basis for collecting the signals in the first place.

## Status and maturity

This is a **reference implementation**, not a validated operational product.

Every accuracy figure produced by this repository — including the Golden Path demo output — is computed on synthetic fixtures. Those numbers validate that the pipeline is wired correctly and replays deterministically. **They are not field performance and must not be reported as detection or identification accuracy.** The demo prints this caveat, and `GoldenSigintEvaluation` carries a `fixture_metrics_only` flag for exactly this reason.

The crate is not published to crates.io (`publish = false`). Use it as a git dependency or vendored source.

## Core capabilities

### Signal understanding

* RF signal classification
* I/Q feature extraction
* temporal analysis and segmentation
* anomaly detection
* open-set recognition
* uncertainty estimation and calibration
* embeddings, similarity and clustering
* sequence analysis

### Dense-spectrum analysis

Real captures frequently contain several simultaneous signals. `DenseSpectrumExtractor` represents a capture as multiple independent components instead of forcing the whole observation into one class.

```text
I/Q capture
   ↓
Hann window → polyphase fold → FFT
   ↓
peak detection over the shifted spectrum
   ↓
component A / component B / component C
```

Each `SignalComponent` reports:

* `center_offset_hz` — signed offset from the capture centre frequency, in `[-fs/2, +fs/2)`
* `bandwidth_hz` — measured −3 dB width around the peak
* `start_sample` / `end_sample` — estimated −3 dB time support, after mixing the component to baseband and filtering to its own bandwidth
* `relative_power_db` and `confidence`, both relative to the strongest component in the capture

Bandwidth and time support are **estimates from a single transform**, not exact signal edges. Components do not currently carry their own embedding; embed the component's band separately if you need one.

## Receiver-aware normalization

A practical fingerprinting system must separate transmitter characteristics from receiver and channel artifacts. `normalize_receiver_capture` corrects a capture against a **supplied** `ReceiverProfile`:

* carrier-frequency-offset bias
* I/Q gain imbalance
* I/Q phase imbalance
* receiver-specific metadata

To be explicit: this applies a known, caller-provided correction. It does **not** estimate those impairments blindly from the signal. You need a characterised receiver. Given one, `cross_receiver_consistency` reduces the risk of treating the same emitter as two identities because the capture hardware changed.

## Open-world emitter discovery

Traditional classifiers assume every signal belongs to a known class. Real RF environments do not.

```text
observation
   ↓
known-class comparison
   ↓
open-set decision
   ↓
unknown observations
   ↓
local clustering
   ↓
provisional emitter hypothesis
```

Repeated unknown observations become a local provisional emitter (`rf-unknown-NNNN`) rather than an undifferentiated `UNKNOWN`. Promotion to `EnrolledLocal` requires both a minimum support count and an explicit `operator_confirmed` flag — `enroll_local` refuses without it. Hypotheses are session-scoped and are not persisted.

## Spectrum representation

A compact, backend-independent spectrum-encoder interface for RF embeddings, masked reconstruction, temporal representation, next-window prediction, few-shot learning and anomaly detection. More advanced learned RF representation models can be integrated behind it without redesigning the rest of the pipeline.

## Evidence graph

A classification result alone is often insufficient. The SDK maintains a session-scoped evidence graph linking observation → signal component → waveform hypothesis → emitter hypothesis → supporting and contradicting evidence, so downstream applications can ask why an identity was proposed, what supports it, what contradicts it, and why the system abstained.

The graph is deliberately local and session-scoped. It is not a persistent global emitter intelligence database.

## Additional ML primitives

Self-supervised and contrastive learning, metric learning, continual and active learning, adaptation, drift detection, ensembles, quantization, pruning, distillation, evaluation, explainability, verification and deterministic inference pipelines.

The objective is not to accumulate model architectures, but to provide reusable RF-specific components that can be tested independently.

## Example applications

Spectrum monitoring, interference investigation, RF anomaly detection, unknown-signal discovery, wireless and industrial RF monitoring, RF ML benchmarking, and research into specific-emitter identification.

Treat the last one as a research direction, not a delivered capability: nothing in this repository has been evaluated against real emitters.

## Integration with simulators

The SDK can consume synthetic observations from external RF simulators such as `spectra-sim`. The simulator is not a runtime dependency and the Golden Path runs from committed fixtures alone.

## Quick start

Requires Rust 1.85 (see `rust-toolchain.toml`).

```bash
git clone https://github.com/nailuJ-dev/anderion-sigint-ml.git
cd anderion-sigint-ml

cargo build --release --locked
cargo test --locked --all-targets --all-features
cargo run --locked --bin sigint-golden-demo
```

`--locked` matters: this project claims deterministic replay, and that claim starts with building from the committed `Cargo.lock`.

### Build profiles

| Profile | Use | `panic` | Symbols |
|---|---|---|---|
| `release` | services, workstations | unwind | kept |
| `release-edge` | embedded / edge targets | abort | stripped |

`release` deliberately unwinds so a panic in one request becomes a `5xx` rather than a dead process, and so crashes leave a usable backtrace. Use `cargo build --profile release-edge` only where a supervisor restarts the process and no post-mortem is expected.

### Optional HTTP service

```bash
cargo build --release --locked --features server --bin anderion-sigint-serve
SIGINT_MODEL_MANIFEST=... SIGINT_MODEL_PAYLOAD=... anderion-sigint-serve
```

Binds `127.0.0.1:8080` by default; `BIND_ADDR` overrides it, and the container image sets `0.0.0.0:8080`. The service has **no authentication and no TLS**. It enforces a request-body limit, a bounded number of concurrent inferences (`503` on overload) and a per-request timeout (`504`). Put it behind an authenticated TLS ingress with rate limiting. See `docs/OPERATIONS.md` and `SECURITY.md`.

### I/Q feature compatibility

Version 0.6 introduced an explicit I/Q feature schema. `IqFeatureSchema::V2FullBandShifted` covers the complete complex baseband using Hann-windowed FFT power bands and is the default. Legacy v0.5 extractors remain available as `IqFeatureSchema::V1NearDc`. Models trained on the legacy near-DC tail must stay on v1 or be retrained and revalidated before moving to v2. See `docs/IQ_FEATURE_SCHEMA.md`.

Since 0.7, captures must contain at least `MIN_IQ_SAMPLES` (16) samples, and `spectrum_bins` may not exceed the capture length. Below those bounds the Hann window destroys the signal and the coarse-band mapping loses its centring guarantee, so the SDK rejects the input instead of returning a plausible-looking zero spectrum.

### Deterministic verification and replay

The verified pipeline binds canonical input, model and configuration digests, ontology and pipeline versions, seed, policy and result into deterministic SHA-256 evidence. Replay distinguishes `Exact`, `DecisionEquivalent` and `NonReproducible`. A divergence in the semantic-consistency outcome is a hard `NonReproducible`, never a downgrade to `DecisionEquivalent`.

`ResultCertificate` carries a `self_digest` over its own fields, recomputed on deserialization, so a certificate cannot be edited field by field. **This is integrity evidence, not a signature**: it does not prove issuer authenticity, sensor authenticity, or that the model was the one you think it was. Anyone able to author a certificate can also compute its digest.

Certificates issued before 0.7 do not replay against 0.7 — the algorithm version and the configuration-digest definition both changed. Re-issue them. See `docs/ONTOLOGY_AND_DETERMINISTIC_VERIFICATION.md`.

## Design principles

* open-world behaviour over forced classification
* calibrated uncertainty over confidence theatre
* deterministic verification where possible
* local evidence over opaque decisions
* modular algorithms over monolithic models
* explicit receiver and channel effects
* reproducible evaluation, honestly labelled

The library forbids `unsafe` code and denies `unwrap`, `expect` and `panic!` at the crate root. Every public type revalidates its invariants on deserialization rather than trusting `#[derive(Deserialize)]` to preserve what its constructor enforced.

## Open-source boundary

This repository provides local RF analysis and ML primitives. It deliberately excludes persistent global emitter identity databases, cross-site strategic correlation, proprietary electromagnetic world models, customer-specific intelligence datasets, and offensive electronic-attack capabilities. See `docs/OPEN_SOURCE_BOUNDARY.md`.

## Contributing

Useful contributions include RF datasets with clear licensing, receiver-domain robustness tests, open-set algorithms, modulation and emitter benchmarks, deterministic feature extractors, calibration methods, evaluation scenarios and reproducible research implementations.

Research-backed pull requests should cite the paper or technical reference that motivated the implementation. See `CONTRIBUTING.md`; report security issues through the process in `SECURITY.md`, not as public issues.

Evaluation against real RF captures is the single most valuable contribution, precisely because the repository currently has none.

## About

`anderion-sigint-ml` is part of the open-source RF sensing initiative developed by **Anderion Systems**. The broader objective is better foundations for software that interprets the electromagnetic environment instead of treating RF as an opaque stream of samples.

**Anderion Systems** — https://anderion-systems.com

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.