# I/Q feature schema compatibility

## v0.5 and earlier: `v1_near_dc`

The legacy reference extractor evaluated only DFT indices `0..K-1` with frequency index `bin/N`. With the default `K = 16`, long captures therefore represented only a narrow near-DC portion of complex baseband.

This behavior is preserved only for compatibility through `IqFeatureSchema::V1NearDc` and `ReferenceIqFeatureExtractor::legacy_v1(...)`.

Serialized extractors created before v0.6 do not contain a schema field. Deserialization deliberately maps that missing field to `V1NearDc`; it never silently applies v0.6 full-band semantics to legacy parameters.

## v0.6 and later: `v2_full_band_shifted`

New extractors use `IqFeatureSchema::V2FullBandShifted` by default:

```text
complex I/Q
  -> Hann window
  -> complex FFT
  -> fftshift
  -> K equal-width coarse power bands across [-Fs/2, +Fs/2)
  -> unit-sum spectral normalization
```

The 12 time-domain scalar features are unchanged. The feature dimension remains `12 + K`, but the numerical meaning of the spectral tail changes.

## Migration

A model trained on v1 spectral features must continue to use v1 features or be retrained/revalidated for v2. Do not feed v2 features into parameters fitted on v1 unless an explicit migration experiment demonstrates equivalence for the intended task.

`center_frequency_hz` remains capture metadata. The v2 spectral tail describes offsets across the complex baseband sampled at `sample_rate_hz`.
