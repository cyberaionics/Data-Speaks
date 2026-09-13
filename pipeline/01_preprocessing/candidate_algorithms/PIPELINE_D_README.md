# Pipeline D — Chebyshev Type II and Robust Statistical Detection

```text
Stage-0 standardized EEG
  → zero-phase Chebyshev Type II 1–40 Hz band-pass
  → local-median / MAD robust statistical artifact detection
  → per-channel recording-level standardization
  → zero-phase FIR decimation to 256 Hz when required
```

## Fixed parameters

The project specification names the Pipeline D method family but does not
provide numeric Chebyshev, robust-detector, or decimation parameters. The
following minimal fixed assumptions are therefore used consistently for all
benchmark recordings:

- Chebyshev Type II IIR: order 4; 40 dB stopband attenuation; 1–40 Hz;
  zero-phase application.
- Artifact detection: per-channel, 0.5-second local median and MAD estimate;
  robust-z threshold 6.0; flagged samples are replaced with their local median.
- Decimation: zero-phase FIR integer-ratio decimation to 256 Hz. CHB02 is
  already 256 Hz, so this stage is explicitly recorded as a no-op rather than
  unnecessarily discarding samples.

The detector retains every channel and time point; only detected sample values
are replaced. All counts are captured in provenance. Pipeline D must be
evaluated under the shared Stage-0, segmentation, labeling, and feature rules;
it is not presumed better than the other candidates.
