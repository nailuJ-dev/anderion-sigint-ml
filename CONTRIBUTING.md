# Contributing

Changes are developed test-first. Add or modify the smallest behavioral test, confirm the test fails for the intended reason, implement the minimal change, then refactor with the suite green.

Branch names use one of these forms:

- `feature/<topic>`
- `fix/<topic>`
- `security/<topic>`
- `perf/<topic>`
- `refactor/<topic>`
- `docs/<topic>`
- `test/<topic>`
- `release/<version>`

Branch names must use the documented semantic prefixes and describe only the software change. Commit subjects should use Conventional Commit style where practical and must not contain development-tool attribution.

Before opening a pull request run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

Security-sensitive parsers and validation changes require negative tests for malformed, oversized, non-finite, tampered and dimension-mismatched input.
