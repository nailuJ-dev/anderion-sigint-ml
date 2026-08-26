# Release Gates

A production tag requires all of the following to pass on the exact revision being released:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
rustup target add wasm32-unknown-unknown
cargo check --lib --no-default-features --target wasm32-unknown-unknown
cargo generate-lockfile
cargo audit
cargo deny check
docker build -t sdk-release-candidate .
```

After `cargo generate-lockfile`, review and commit `Cargo.lock` before producing a binary/container release, then rebuild with locked dependencies.

Artifact/model bundles must pass the SDK's manifest schema, payload-length, size-limit, allowed-model-type and SHA-256 integrity checks before deserialization and use.

## Ontology and deterministic-verification gate

```bash
cargo test --test ontology_verification
```

A release must also confirm that the replay tests detect changed model/configuration context, ontology contradictions are surfaced deterministically, and no new non-public or remote dependency has been introduced.


## Golden Path smoke test

The release is not publishable unless the first-user workflow runs end-to-end:

```bash
cargo run --quiet --bin sigint-golden-demo
```

The generated report must include an exact deterministic replay. Fixture metrics are integration evidence only, not field-performance claims.
