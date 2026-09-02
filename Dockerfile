# syntax=docker/dockerfile:1.7

# Pin by digest for reproducible builds. Refresh with:
#   docker buildx imagetools inspect rust:1.85-bookworm --format '{{.Manifest.Digest}}'
#   docker buildx imagetools inspect gcr.io/distroless/cc-debian12:nonroot --format '{{.Manifest.Digest}}'
# then set RUST_IMAGE=rust:1.85-bookworm@sha256:... in CI.
ARG RUST_IMAGE=rust:1.85-bookworm
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian12:nonroot

FROM ${RUST_IMAGE} AS builder
WORKDIR /src
COPY . .
# `--locked` refuses to resolve away from the committed Cargo.lock.
# Cache mounts keep the dependency graph warm without a stub-crate dance; the
# binary is copied out because cache mounts do not persist into the image.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --release --locked --features server --bin anderion-sigint-serve \
 && mkdir -p /out \
 && cp target/release/anderion-sigint-serve /out/anderion-sigint-serve

FROM ${RUNTIME_IMAGE}
COPY --from=builder /out/anderion-sigint-serve /usr/local/bin/anderion-sigint-serve
USER nonroot:nonroot
# The service has no built-in authentication or TLS. Terminate both at an
# authenticated ingress and apply the limits described in docs/OPERATIONS.md.
ENV BIND_ADDR=0.0.0.0:8080
# SIGINT_MODEL_MANIFEST and SIGINT_MODEL_PAYLOAD are mandatory and deliberately
# not baked in: mount the verified manifest/payload pair at runtime.
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/anderion-sigint-serve"]
