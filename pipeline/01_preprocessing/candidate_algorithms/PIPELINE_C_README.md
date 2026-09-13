# Pipeline C — Butterworth, ASR, ICA, Standardization

## Purpose

Pipeline C is the subspace-based preprocessing candidate for CHB-MIT EEG:

```text
Stage-0 standardized EEG
  → zero-phase Butterworth 1–40 Hz band-pass
  → Artifact Subspace Reconstruction (ASR)
  → FastICA artifact removal
  → per-channel, recording-level standardization
  → 256 Hz polyphase resampling when needed
```

It must be compared under the repository's shared Stage-0, windowing,
labeling, feature, and split policies. It is not assumed to be superior to
Pipelines A or B.

## ASR policy

ASR is applied after the 1 Hz high-pass component of the Butterworth filter,
as required for stable calibration. It uses `meegkit.asr.ASR` with a
conservative cutoff of 5.0 and Euclidean covariance processing.

Each recording is calibrated separately from its first 60 seconds. The ASR
implementation robustly selects clean sub-windows during calibration; the
provenance records the fraction retained. This avoids training an ASR model
across patients, recordings, or channel montages.

Because an aggressive ASR setting can suppress seizure-relevant structure,
the cutoff and calibration rule are fixed before benchmark comparison and must
be reported with every result.

## ICA policy

Pipeline C reuses the deterministic FastICA policy from Pipeline A: up to 15
components, `random_state=42`, `FP1-F7` as an EOG proxy, correlation threshold
0.4, and muscle threshold 0.3. Automatic component exclusion is heuristic and
the excluded indices are recorded in provenance.

## Validation

Run the focused ASR check and end-to-end check from the repository root:

```bash
MPLCONFIGDIR=/private/tmp/data_speaks_mpl .venv/bin/python notebooks/41_test_asr.py
MPLCONFIGDIR=/private/tmp/data_speaks_mpl .venv/bin/python notebooks/41_test_pipeline_c.py
```

These checks use CHB02 recordings already present locally and verify finite
output, preserved channels and sampling rate, shape preservation, and the
final per-channel zero-mean/unit-standard-deviation invariant.
