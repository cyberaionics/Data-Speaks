# Data-Speaks

EEG seizure-detection experiments based on the CHB-MIT Scalp EEG Database.
The repository reproduces a preprocessing and feature pipeline, then compares
several classifiers under leave-one-record-out evaluation.

This README documents how the raw data is interpreted, why each transformation
is applied, how predictions are evaluated, and what the current Patient 1
results mean.

## Scope

The current implementation is an offline, patient-level inference experiment.
It answers this question:

> Given labeled recordings from one patient, how well can a model predict the
> held-out recording when trained on the patient's other recordings?

It is not yet a streaming clinical detector or a patient-independent model.
The term `inference` means prediction on a held-out EDF record after the shared
preprocessing and feature-extraction steps.

## Environment

The project uses Python 3.12 or newer and `uv` for dependency management.
Using `uv run` selects the project environment.

```bash
uv sync
```

The main dependencies are MNE, NumPy, SciPy, scikit-learn, and Matplotlib.

## Data Layout

Patient data is expected in this layout:

```text
data/raw/physionet.org/
	chb01/
		chb01_01.edf
		chb01_03.edf.seizures
		chb01-summary.txt
```

The EDF files contain multichannel scalp EEG. The patient summary contains
the authoritative seizure start and end times for each EDF. PhysioNet also
provides binary `.edf.seizures` files; those files are retained as source data,
but this project reads the human-readable summary because its timing fields
are directly usable and auditable.

Raw EDF and seizure files are ignored by Git because they are large source
data, not source code. Do not commit them to the repository.

## How the Raw Data Is Interpreted

### 1. EDF channels

Each EDF is loaded with MNE. The loader removes channels whose names indicate
non-EEG signals, including ECG, EKG, VNS, stimulation, pulse, respiration,
temperature, EMG, and EOG channels.

The remaining channels are matched against this standard bipolar montage, in
this order:

```text
FP1-F7, F7-T7, T7-P7, P7-O1,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2,
FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ, CZ-PZ
```

The reason for a fixed montage is to make feature columns comparable between
recordings and between team algorithms. Channel names are normalized for case
and whitespace before matching.

The loader requires at least 10 of the 18 target channels. It does not invent
missing channels, preventing fabricated signals from being treated as EEG.

CHB01 contains a duplicate `T8-P8` channel in some EDFs. MNE disambiguates the
duplicate, and the current matching retains the first matching channel.
Consequently, processed CHB01 data contains 17 channels rather than 18. This
is why reporting code infers channel count from feature width instead of
assuming 18.

### 2. Filtering

Pipeline A applies a zero-double-phase FIR bandpass filter:

```text
low cutoff:  0.5 Hz
high cutoff: 45 Hz, limited to 95% of Nyquist frequency
window:      Hamming
design:      FIR windowed filter
```

The low cutoff removes slow baseline drift. The high cutoff removes high-
frequency activity outside the intended EEG band while preserving common
seizure-related activity. Zero-double phase reduces phase shift in the result.

### 3. Artifact removal

ICA is fitted with FastICA, up to 20 components, random seed 42, and a maximum
of 500 iterations. Components identified from frontal channels as EOG-like or
as muscle-like are candidates for exclusion. At most half of the fitted
components are removed.

ICA failures are recorded in provenance and the pipeline continues with the
unmodified signal. This keeps one problematic recording from stopping a full
patient run, but metadata should be checked before treating every record as
equally clean.

The CHB-MIT files do not always contain sensor positions. MNE therefore warns
that muscle-component scoring may rely only on slope information. This is a
known limitation of the artifact interpretation.

### 4. Per-channel normalization

Each retained channel is normalized across the full recording:

```text
normalized = (signal - channel_mean) / channel_standard_deviation
```

This reduces amplitude-scale differences between channels and recordings
before feature computation. A small epsilon prevents division by zero for a
constant channel. Means and standard deviations are saved in provenance.

### 5. Sampling rate

Signals are resampled to 256 Hz when their original sampling rate differs.
CHB-MIT CHB01 recordings are already 256 Hz, so no resampling is normally
needed for Patient 1. A fixed rate is required because epoch lengths and
frequency bins depend on the sampling rate.

## Segmentation and Labels

### Epochs and windows

The normalized signal is split into non-overlapping 2-second epochs. Three
consecutive epochs are stacked into one 6-second classification window. The
configured overlap is 50%, but epochs are discrete, so the implementation
rounds the step to 2 epochs. Adjacent windows therefore overlap by 1 epoch,
or about 33.3% in the current implementation.

For CHB01, each retained window has this conceptual shape:

```text
(stacked epochs, channels, samples) = (3, 17, 512)
```

The window start time is the start of the first 2-second epoch in the stack.

### Seizure annotation

For each EDF, the parser first supports a text `.seizure` sidecar. When that
does not exist, it reads the matching block from `<patient>-summary.txt` and
extracts `Seizure Start Time` and `Seizure End Time` pairs.

This fallback is necessary because the CHB-MIT distribution provides binary
`.edf.seizures` sidecars while the summary contains readable timings.

### Window labels

For each 6-second window:

- `1` (`ictal`) when at least 50% of the window overlaps a seizure interval.
- `0` (`interictal`) when it does not overlap a seizure and is at least 30
	seconds away from the nearest seizure boundary.
- `-1` (`unknown`) for boundary windows too close to a seizure to assign
	confidently.

Unknown windows are removed from `X`, `y`, and timestamps before model
training. Keeping these arrays aligned is essential: each label must refer to
the same feature row and start time.

## Features

The preprocessing entry point currently writes spectral features only. For
each of the three stacked epochs, it computes FFT power in eight equally
spaced bands from 0.5 to 25 Hz. Power is summed separately for every channel
and band.

For CHB01 this produces:

```text
3 stacked epochs x 17 channels x 8 bands = 408 features per window
```

The repository also contains optional Hjorth and basic-statistics extractors,
but they are not included in the default `preprocess_patient.py` feature set.

## Inference and Evaluation

The classifier evaluation uses leave-one-record-out (LOSO):

1. Hold out one EDF recording.
2. Train on all other recordings from the same patient.
3. Predict every labeled window in the held-out recording.
4. Convert positive windows into detection events.
5. Repeat for every recording.

This prevents windows from the held-out recording appearing in its training
data. It does not measure generalization to a new patient; that requires a
patient-level holdout protocol.

Available models:

- SVM: RBF kernel, `C=1`, `gamma=0.1`, balanced class weights.
- Logistic Regression: balanced classes, maximum 1000 iterations.
- Random Forest: 100 trees, maximum depth 12, balanced subsampling.
- KNN: 5 neighbors.

All model adapters standardize features before fitting. The same train/predict
interface ensures that algorithms receive identical folds, features, labels,
and event rules.

Run Patient 1 preprocessing and inference:

```bash
uv run python scripts/preprocess_patient.py --patient chb01
uv run python scripts/run_patient.py --patient chb01 --model random_forest
```

Compare the faster baseline models:

```bash
uv run python scripts/run_patient.py --patient chb01 --model logistic_regression
uv run python scripts/run_patient.py --patient chb01 --model random_forest
uv run python scripts/run_patient.py --patient chb01 --model knn
uv run python scripts/compare_models.py --patient chb01
```

The RBF SVM can be run with `--model svm`, but it is substantially slower on
the current 36,000-plus-window feature set. Run it deliberately rather than
as part of every comparison.

## Event Metrics

Window predictions become events when a positive prediction follows a negative
prediction. A 30-second refractory period prevents multiple events from being
counted too close together.

Reported metrics:

- **Sensitivity:** whether at least one predicted event falls within 5 seconds
	before or after an annotated seizure interval for a seizure-containing
	recording.
- **Latency:** time from the earliest annotated seizure onset to the first
	qualifying predicted event.
- **False detections per 24 hours:** event count normalized by recording
	duration for recordings without annotated seizures.

Sensitivity is currently summarized per seizure-containing recording, not per
individual seizure. This is acceptable for the one-seizure-per-record pattern
in CHB01, but should be upgraded to seizure-level sensitivity for patients
with multiple seizures in one EDF.

Because ictal windows are rare, accuracy is not the primary selection metric.
A model that predicts almost everything as interictal can have high accuracy
while missing seizures.

## Current Patient 1 Interpretation

The available CHB01 data contains 42 EDF recordings. Seven contain seizure
annotations and 35 do not. After preprocessing and removal of boundary
windows, the generated dataset contains:

```text
total windows:       36,343
ictal windows:          112
interictal windows:  36,231
retained channels:       17
features/window:        408
```

Current completed LOSO results are:

| Model | Seizure-record sensitivity | Mean latency (s) | Mean false detections/24h |
|---|---:|---:|---:|
| Random Forest | 1.000 | 5.86 | 2.06 |
| Logistic Regression | 1.000 | 1.00 | 18.07 |
| KNN | 0.857 | 2.17 | 1.37 |

These are Patient 1 experiment results, not a clinical performance claim.
Random Forest is the current best balance because it retains the same
record-level sensitivity as Logistic Regression with substantially fewer false
detections. KNN has the lowest false-detection rate but misses one of the seven
seizure-containing records.

## Outputs

Processed files are written to:

```text
data/processed/<patient>/
	<record>_X.npy
	<record>_y.npy
	<record>_starts.npy
	<record>_meta.json
```

Benchmark summaries and per-record metrics are written to:

```text
results/benchmarks/<model>_<patient>_loso.json
results/benchmarks/comparison_<patient>.json
```

Window-level predictions are written to:

```text
results/predictions/<model>/<patient>/<record>_y_pred.npy
```

PCA and spectral-shift reports are written to `results/figures/` by:

```bash
uv run python scripts/math_report.py --patient chb01
```

Prediction-focused interpretation figures are generated with:

```bash
uv run python scripts/plot_results.py --patient chb01 --model logistic_regression
```

This creates:

- `<patient>_class_balance.png`: total and per-record ictal/interictal windows.
- `<patient>_model_comparison.png`: sensitivity, false detections, and latency
	across completed model benchmarks.
- `<patient>_<model>_confusion_matrix.png`: aggregate window predictions where
	saved prediction arrays are available.
- `<patient>_<model>_<record>_timeline.png`: true labels, predictions, and
	annotated seizure intervals over time.

Use `--record chb01_03` to select a particular seizure-containing recording
for the timeline plot.

## Reproducibility and Team Comparison

Team algorithms are comparable only when they use the same:

1. Patient and EDF recording set.
2. Channel selection and preprocessing.
3. Window and label definitions.
4. LOSO folds.
5. Event refractory period and timing tolerance.
6. Metric aggregation rules.

Each team submission should provide one JSON benchmark with the model name,
patient, summary metrics, and per-record results. Choose a model in this order:

1. Reject models with materially lower seizure sensitivity.
2. Among models with comparable sensitivity, prefer fewer false detections.
3. Use latency as a tie-breaker.

Do not compare a single favorable recording, raw window accuracy, or results
from different preprocessing pipelines as if they were equivalent.

## Known Limitations

- The current experiment is patient-specific rather than patient-independent.
- CHB01 has a duplicate channel name and therefore uses 17 retained channels.
- ICA quality is limited when sensor positions are absent.
- The default spectral features are raw FFT power sums, not normalized band
	power or log power.
- Current sensitivity is record-level and should become seizure-level for
	multi-seizure records.
- The RBF SVM is computationally expensive for the current feature volume.
- The evaluation runner expects labeled recordings. A production unlabeled
	inference path would need a separate prediction-only command that does not
	manufacture interictal labels from missing annotations.