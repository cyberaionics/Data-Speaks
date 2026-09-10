# Stage 01 — Preprocessing

Common Stage 0 (baseline) plus candidate pipelines A/B/C/D.

## Common Stage 0 (immutable)
- Read EDF
- Drop non-EEG channels (ECG, VNS, aux)
- Restrict to the standard 18 bipolar channels

## Candidate Pipeline A — FIR → ICA → Z-score → 256 Hz
See `candidate_algorithms/PIPELINE_A_README.md`.

## I/O contract
- Input : path to a raw EDF
- Output: `(n_channels, n_samples)` float32, 256 Hz, Z-scored
- Every step returns a `provenance` dict