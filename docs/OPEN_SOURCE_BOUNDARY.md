# Standalone Boundary

This repository is a complete standalone SDK. Its default build has no dependency on non-public crates, remote inference services, remote model registries, hidden datasets, external feature-generation systems or undisclosed model formats.

The caller supplies numerical observations and, when using extension traits, an implementation directly in its own process. The SDK never auto-discovers or downloads an implementation.

Out of scope by design:

- signal acquisition and interception;
- communications-content decoding;
- protocol exploitation;
- emitter control, jamming or waveform manipulation;
- automatic network egress from the ML runtime.

The public release boundary is therefore defined by functionality, not by a connection to another codebase.

The ontology and deterministic-verification layers are implemented entirely in this repository. They do not resolve schemas, graphs, model metadata or verification evidence through a private or remote service.

