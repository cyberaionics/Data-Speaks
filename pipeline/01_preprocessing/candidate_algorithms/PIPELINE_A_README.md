# Pipeline A — Conservative / Classical Reference Pipeline

> **Status:** candidate — to be benchmarked against Pipelines B, C, D under the
> shared evaluation protocol. Nothing here claims optimality in advance.

## Definition

```
FIR band-pass (1-40 Hz)
    -> ICA artifact removal
    -> per-channel Z-score normalization
    -> resample to 256 Hz
```

**Purpose:** conventional, interpretable reference. Every choice is the
established default in its category, so Pipeline A anchors the comparison —
if a fancier pipeline cannot beat this one under the common protocol, the
added complexity is not justified.

## Input / Output Contract

| | |
|---|---|
| Input | Standardized EEG (`mne.io.Raw`) from **Common Stage 0** + `metadata.csv` row |
| Output | Processed `Raw` (256 Hz, Z-scored) + provenance `dict` |
| Output format | **FIF** (`raw.save`) + JSON provenance — **not EDF** |

### Why FIF and not EDF for processed output?

EDF stores integer digital samples with physical min/max scaling. After
Z-score normalization the signal is floats near ±5; EDF's 16-bit
quantization would silently destroy precision exactly where downstream
features need it. FIF preserves float64 exactly. Raw EDFs remain untouched
in `data/raw/` (rule #1).

## Parameter Table (every value is explicit, auditable, versioned)

| Step | Parameter | Value | Justification |
|---|---|---|---|
| Filtering | design | `firwin`, zero-phase | linear phase, no seizure-morphology distortion |
| Filtering | band | 1-40 Hz | 1 Hz: drift/DC; 40 Hz: 60 Hz line noise, muscle |
| Filtering | notch | none | 60 Hz > 40 Hz cutoff; band-pass already suppresses it |
| ICA | algorithm | FastICA | established, well-characterized |
| ICA | n_components | 15 (clamped to rank-1) | fewer sources than channels; standard EEG practice |
| ICA | random_state | 42 | reproducibility |
| ICA | eye detection | `find_bads_eog(ch_name="Fp1")`, thresh 0.4 | **proxy** — Stage 0 removed EOG/ECG |
| ICA | muscle detection | `find_bads_muscle`, thresh 0.3 | spectral high-frequency signature |
| Normalize | scheme | per-recording Z-score | mean 0, unit variance per channel |
| Normalize | window option | NOT used | per-window stats would leak across windows |
| Resample | target | 256 Hz | common protocol; CHB-MIT native rate -> usually no-op |
| Resample | anti-alias | polyphase FIR (MNE default, kept ON) | naive decimation would fold high-freq energy |

## Known Limitations (to be examined in evaluation)

1. **Fp1-as-EOG is a heuristic.** Without true EOG channels, blink labeling
   can miss lateral eye movements or falsely reject frontal seizure
   activity. All IC scores are logged to provenance so Level-1 evaluation
   can audit rejections — if Pipeline A rejects components correlated with
   the seizure interval, that must be reported, not hidden.
2. **No explicit muscle/EMG channel check beyond the spectral detector.**
3. **Zero-phase FIR pre-ringing** near sharp transients is a known
   trade-off of non-causal filtering; accepted because analysis is offline.
4. **Recording-level Z-score** means a channel dominated by one long
   artifact still ends with unit variance. Stage-0 QC flags should catch
   such recordings before they reach this pipeline.

## Files

```
candidate_algorithms/
├── filtering/fir_bandpass.py           # substep 1
├── artifact_removal/ica.py             # substep 2 (fit / select / apply)
├── normalization/zscore.py             # substep 3
├── resampling/resample_256.py          # substep 4
└── pipeline_a.py                       # orchestrator + PipelineAConfig
```

Files are named after the algorithm they implement, per the current team
naming convention. Substitute files import cleanly from the substep modules,
so the shared evaluator only ever needs
`run_pipeline_a(raw_std, config, recording_id)`.

## Running on the benchmark

1. Obtain `raw_std` **only** via the shared Stage-0 loader — never read the
   raw EDF directly inside Pipeline A (that would bypass channel policy/QC
   and invalidate the comparison).
2. Loop over the ~6 benchmark recordings; save each `raw_out` to
   `data/processed/pipeline_A/<recording_id>_raw.fif`.
3. Write all provenance dicts to `results/benchmarks/pipeline_A_provenance.json`
   for the shared evaluator.
4. Record `preprocessing_method = pipeline_A_fir_ica_zscore_256` in
   `metadata.csv`.

## Do NOT

- reorder substeps (ICA operates on native-rate filtered data by contract),
- disable anti-aliasing during resampling,
- feed raw EDFs directly into the pipeline,
- change parameters without a commit message documenting why —
  the shared evaluator treats the config dataclass as the ground truth.
