# Golden Path: raw I/Q to verified classification

The Golden Path is the shortest supported route from a bounded raw I/Q capture to a reproducible SDK result. It is deliberately self-contained: no remote model, private dataset, network service, or non-public dependency is required.

## Run it

```bash
./scripts/run_golden_path.sh
```

Or, with Docker and no local Rust toolchain:

```bash
./scripts/run_golden_path_docker.sh
```

The command:

1. loads the bundled labeled I/Q training fixtures;
2. extracts deterministic time-domain and coarse spectral features;
3. fits the public hash-projection encoder, prototype classifier, and anomaly detector;
4. classifies `demo-data/golden-path/scenario.json`;
5. maps the result into the public ontology;
6. applies deterministic verification;
7. replays the inference and verifies exact reproducibility;
8. evaluates the small held-out fixture set;
9. writes `artifacts/golden-path/result.json`.

To use another capture, keep the same JSON schema as `scenario.json` and run:

```bash
./scripts/run_golden_path.sh --scenario path/to/your-scenario.json
```

## Input contract

A capture contains an id, timestamp, sample rate, center frequency and at most 16,384 finite complex samples. The reference feature extractor emits a fixed-size vector containing amplitude/power statistics, zero-crossing/phase statistics and a normalized coarse spectrum.

## What this proves

It proves that raw I/Q data can traverse the public SDK end-to-end and produce a repeatable prediction, ontology consistency report and deterministic certificate.

It does **not** prove field accuracy. The bundled signals are synthetic reference fixtures. Replace them with a representative, legally obtained and properly split dataset before making operational performance claims.
