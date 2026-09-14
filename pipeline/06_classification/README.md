# Stage 06: Classification + Standalone Inference

## Overview

Stage 06 is the final supervised machine-learning stage of the Data-Speaks EEG pipeline. It evaluates multiple binary classifiers for distinguishing **ictal (seizure)** and **interictal** EEG windows, then saves a selected model and its preprocessing artifacts for standalone inference on an EDF recording.

The classification stage operates on the representation produced by the earlier pipeline stages:

```text
Raw EDF
  ↓
Stage 0: standardization + QC
  ↓
Pipeline C: Butterworth → ASR → FastICA → z-score
  ↓
Stage 02: 4 s windows, 50% overlap
  ↓
Stage 03: 442 EEG features/window
  ↓
Stage 06: scaling → PCA → classifier
  ↓
Window-level predictions
```

> **Scope:** This stage performs binary classification and standalone inference. A separate regression task is not part of the current project.

---

## Key Design Principles

### 1. Recording-wise leakage prevention

Windows from the same recording are never placed in both training and test sets within a fold.

**GroupKFold** is performed using `recording_id`.

This is important because the segmentation uses 4-second windows with 2-second stride, so neighboring windows overlap by 50%.

### 2. Per-fold preprocessing

For every cross-validation fold:

```text
training recordings
      ↓
StandardScaler.fit()
      ↓
PCA.fit()
      ↓
classifier.fit()

held-out recordings
      ↓
StandardScaler.transform()
      ↓
PCA.transform()
      ↓
classifier.predict()
```

The scaler and PCA are therefore fitted only on the training portion of each fold.

### 3. Labels are not used for scaling or PCA

Labels are used as the supervised target for classification and for post-hoc evaluation.

Ambiguous windows (`label = -1`) are excluded before classification.

### 4. Primary evaluation metric

Because CHB01 is extremely imbalanced, **PR-AUC (Average Precision)** is used as the primary model-selection metric.

Accuracy is reported, but it is not used to select the final model.

---

## Classification Input

The existing CHB01 feature table is:

```text
results/tables/chb01_all_features.csv
```

It contains:

- **72,951 windows**
- **442 numerical EEG features**
- **7 metadata columns**

Metadata:

```text
patient_id
recording_id
window_id
start_sec
end_sec
duration_sec
label
```

Labels:

```text
0  = interictal
1  = ictal
-1 = ambiguous
```

Ambiguous windows are excluded from classification.

---

## Models

The final computationally practical implementation evaluates four classifiers:

| Model | Configuration |
|---|---|
| **Linear SVM** | `LinearSVC(C=1.0, class_weight="balanced")` |
| **Logistic Regression** | `class_weight="balanced", max_iter=1000` |
| **Random Forest** | `n_estimators=100, class_weight="balanced"` |
| **KNN** | `n_neighbors=5, metric="euclidean"` |

### Note on SVM

An earlier draft used an RBF-kernel SVM. On the full CHB01 feature matrix (~73k windows × 110 PCA dimensions), that configuration was computationally impractical.

The final implementation therefore uses a **Linear SVM** so that the complete grouped evaluation can run in a reasonable amount of time.

This is a methodological change and should be stated explicitly when comparing results with an RBF-SVM implementation.

---

## Dimensionality Reduction inside Classification

The Stage 04 exploratory PCA was fitted on the entire cohort for exploratory analysis.

For **classification**, that global PCA is not reused.

Instead, each GroupKFold training split independently fits:

```text
442 features
   ↓
StandardScaler
   ↓
PCA(n_components=110)
   ↓
classifier
```

The held-out recording is transformed using those fitted objects.

This prevents test-fold information from entering the PCA basis.

---

## Evaluation

### Cross-validation

- **5-fold GroupKFold**
- grouping variable: `recording_id`
- 4 classifiers evaluated on the same folds

Metrics:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- PR-AUC / Average Precision
- TP, TN, FP, FN

For folds containing no ictal windows, positive-class metrics that are mathematically undefined are stored as `NaN` and excluded from their summary means.

---

## CHB01 Classification Results

The completed grouped evaluation produced the following mean results:

| Model | PR-AUC | F1 | Recall / Sensitivity | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5725 | 0.4327 | 0.6548 | 0.9597 | 0.9961 |
| Logistic Regression | 0.5730 | 0.4393 | 0.6592 | **0.9653** | 0.9961 |
| **Random Forest** | **0.6310** | 0.2858 | 0.2065 | 0.9467 | 0.9974 |
| KNN | 0.5702 | **0.5390** | 0.4622 | 0.8132 | **0.9984** |

### Model selection

The final model is selected by **mean PR-AUC**.

Therefore:

```text
Selected model: Random Forest
Mean PR-AUC:    0.6310
```

This means Random Forest was selected for the final inference artifact because it achieved the highest mean PR-AUC among the evaluated models.

This does **not** mean Random Forest was best on every metric:

- Logistic Regression had the highest ROC-AUC.
- KNN had the highest F1 and accuracy.
- Random Forest had the highest PR-AUC.

---

## Saved Classification Outputs

### Metrics

```text
results/tables/chb01_classification_fold_metrics.csv
results/tables/chb01_classification_metrics.csv
```

### Figures

```text
results/figures/classification/model_comparison.png
results/figures/classification/confusion_matrices.png
```

### Final model artifacts

```text
models/chb01_final_model/
├── scaler.pkl
├── pca.pkl
├── classifier.pkl
├── feature_cols.pkl
└── metadata.json
```

The saved final model was trained using all eligible CHB01 recordings **except `chb01_03`**, which was reserved for the standalone inference demonstration.

---

## Standalone Inference

### What inference means here

Inference means applying the already-fitted pipeline to an EDF recording without refitting the model.

```text
New EDF
   ↓
Stage 0
   ↓
Pipeline C
   ↓
4 s segmentation
   ↓
442 feature extraction
   ↓
saved StandardScaler
   ↓
saved PCA
   ↓
saved classifier
   ↓
window-level predictions
```

The inference path does **not require seizure annotations** to generate predictions.

Annotations may optionally be supplied afterward for evaluation.

---

## Inference Artifacts

Inference loads:

```text
models/chb01_final_model/scaler.pkl
models/chb01_final_model/pca.pkl
models/chb01_final_model/classifier.pkl
models/chb01_final_model/feature_cols.pkl
models/chb01_final_model/metadata.json
```

These objects are loaded from disk and are **not refitted during inference**.

The exact 442-feature ordering used during training is stored in `feature_cols.pkl`.

---

## Inference Validation: chb01_03

`chb01_03` was excluded from final-model training and then processed as the held-out inference demonstration.

Command:

```powershell
python pipeline/06_classification/inference.py --recording chb01_03
```

Input:

```text
Recording: chb01_03
Windows:   1,799
```

Prediction output:

```text
Predicted interictal: 1,799
Predicted ictal:         0
```

Maximum predicted ictal probability:

```text
0.38
```

Several of the highest-scoring windows occurred within the known seizure interval:

```text
3022–3026 s : 0.38
3006–3010 s : 0.34
3000–3004 s : 0.27
3024–3028 s : 0.18
3016–3020 s : 0.17
```

The held-out recording therefore received higher ictal probability scores around the annotated seizure, but no window crossed the default 0.5 decision threshold.

### Interpretation

The inference pipeline completed successfully end-to-end.

The held-out `chb01_03` result indicates that the selected Random Forest produced some ranking/discriminative signal around the seizure interval, but its default binary threshold produced no positive predictions for this recording.

This should be reported as a **generalization limitation**, not as successful seizure detection.

---

## Running

### Classification + model saving

```powershell
python -u pipeline/06_classification/run_classification_chb01.py
```

This:

1. Loads `chb01_all_features.csv`
2. Removes ambiguous windows
3. Runs 5-fold GroupKFold
4. Fits scaler + PCA + classifier inside each fold
5. Saves fold and summary metrics
6. Selects the model using mean PR-AUC
7. Fits the selected final model excluding `chb01_03`
8. Saves scaler, PCA, classifier, and feature schema

### Held-out inference demonstration

```powershell
python pipeline/06_classification/inference.py --recording chb01_03
```

### New EDF

```powershell
python pipeline/06_classification/inference.py ^
    --edf path\to\new_recording.edf ^
    --recording new_recording_id ^
    --patient chb01 ^
    --no-validate
```

---

## Files

```text
pipeline/06_classification/
├── classification.py
├── run_classification_chb01.py
├── inference.py
└── README.md
```

### `classification.py`

Contains:

- classifier definitions
- feature loading
- ambiguous-label filtering
- grouped cross-validation
- per-fold scaling
- per-fold PCA
- metric calculation
- model selection
- final model fitting

### `run_classification_chb01.py`

Runs the CHB01 evaluation and saves the final inference artifacts.

### `inference.py`

Runs the full standalone EDF inference path without requiring annotations.

---

## Limitations

1. **Extreme class imbalance**

CHB01 contains approximately:

```text
99.70% interictal
0.30% ictal
```

Therefore accuracy alone is not informative.

2. **Single-patient cohort**

The current analysis is limited to CHB01. Results should not be interpreted as generalization across the full CHB-MIT population.

3. **Overlapping windows**

The 50% window overlap creates strong dependence between adjacent windows. Recording-wise grouping is therefore essential for evaluation.

4. **Inference generalization**

The held-out `chb01_03` demonstration produced no binary ictal predictions at the 0.5 threshold despite assigning higher probabilities to several seizure-period windows.

5. **No clinical claim**

This is an educational/research-oriented computational study and does not establish clinical diagnostic performance.

---
