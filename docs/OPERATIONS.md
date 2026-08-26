# Operations manual

## Build and test

Required toolchain: Rust 1.85 or newer.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

Before producing a release binary, generate and commit a `Cargo.lock` in the release branch, then build with `--locked`.

## Model artifact workflow

1. Serialize a reference model bundle to JSON.
2. Create an `ArtifactManifest` from the exact payload bytes.
3. Store payload and manifest read-only.
4. At process startup, use `load_verified_payload` / `load_reference_bundle`.
5. Reject startup on schema, type, length, digest or model-invariant failure.

Do not overwrite a model payload in place. Publish a new immutable path/version and roll replicas.

## Service deployment

The reference service exposes:

- `GET /healthz` → `204`;
- `POST /v1/predict` → JSON `Prediction`.

Request bodies are capped at 256 KiB. The process does not provide TLS or authentication; terminate TLS and enforce identity/rate limits at the ingress. Mount model files read-only and run the image with a read-only root filesystem, no added Linux capabilities and `no-new-privileges` where supported.

## Resource sizing

Feature vectors are capped at 65,536 values, embeddings at 8,192 values, batch calls at 65,536 observations, and reference artifacts at the policy byte limit. Tighten these limits per deployment when the expected model shape is smaller.

## Rollback

Keep the previous model manifest/payload pair immutable. Rollback is performed by changing the deployment's model paths to the prior pair and restarting/rolling the stateless replicas.
