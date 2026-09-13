# Data-Speaks Final Pipeline Comparison

## 1. Executive Summary

This report compares the verified outputs of preprocessing Pipelines A, B, C,
and D on the current CHB02 scope. The underlying evaluation results were read
from `results/benchmarks/level1_evaluation.json` and
`results/benchmarks/common_evaluation.json`; no preprocessing or evaluation was
rerun for this report.

Pipeline B has the strongest measured combination in this limited Level 1 test:
it has the highest synthetic-noise SNR improvement (4.173356 dB), highest
channel correlation (0.861298), and lowest recorded clean runtime (0.022398 s).
Pipeline D also improves synthetic-noise SNR (3.862070 dB), but its gamma-band
PSD preservation ratio is 0.000213, which requires scientific caution. Pipeline
C has negative measured SNR improvement (-2.657980 dB) and the highest runtime
in this test. These observations do not establish a global winner: Level 1 uses
one 60-second segment, and downstream classification is preliminary
recording-wise evaluation from one patient.

## 2. Preprocessing Performance Table

All Level 1 values below are measured on `chb02_01`, using the same 60-second
segment, 23 channels, seed 42, 5% synthetic noise scale, Welch PSD calculation,
and 1-40 Hz frequency bands. Runtime is reported for the clean input; noisy-input
runtime is included separately.

### PSD preservation ratio

A ratio near 1 indicates similar relative band-power proportion to the raw
reference. Values are band-specific and should not be replaced by an overall
winner score.

| Frequency band | Pipeline A | Pipeline B | Pipeline C | Pipeline D |
|---|---:|---:|---:|---:|
| Delta, 1-4 Hz | 0.981987 | 1.001949 | 0.902723 | 0.793568 |
| Theta, 4-8 Hz | 1.374482 | 1.080371 | 1.420868 | 1.991231 |
| Alpha, 8-13 Hz | 1.313313 | 1.020022 | 1.548353 | 1.990724 |
| Beta, 13-30 Hz | 1.016120 | 0.932736 | 1.868936 | 1.034690 |
| Gamma, 30-40 Hz | 1.046102 | 0.510955 | 1.197539 | 0.000213 |

### Other Level 1 metrics

| Metric | Pipeline A | Pipeline B | Pipeline C | Pipeline D |
|---|---:|---:|---:|---:|
| SNR improvement (dB) | 0.405667 | 4.173356 | -2.657980 | 3.862070 |
| Channel correlation | 0.691284 | 0.861298 | 0.754035 | 0.677441 |
| Clean runtime (s) | 0.093832 | 0.022398 | 0.379530 | 0.029841 |
| Noisy runtime (s) | 0.066930 | 0.020237 | 0.401407 | 0.028609 |
| Peak memory (MB) | 828.875 | 834.719 | 846.516 | 846.594 |

## 3. Downstream Performance Table

PCA and clustering use the common workflow recorded in
`common_evaluation.json`: standardized PCA targeting 95% explained variance,
and KMeans with two clusters, `n_init=10`, and seed 42. Classification uses the
same imputation, standardization, balanced logistic regression, hyperparameters,
and held-out recording (`chb02_19`) for all pipelines.

| Metric | Pipeline A | Pipeline B | Pipeline C | Pipeline D |
|---|---:|---:|---:|---:|
| PCA component count | 142 | 128 | 130 | 164 |
| PCA explained variance | 0.950486 | 0.950447 | 0.950556 | 0.950548 |
| Clustering algorithm | KMeans | KMeans | KMeans | KMeans |
| Number of clusters | 2 | 2 | 2 | 2 |
| Silhouette score | 0.293033 | 0.477125 | 0.445212 | 0.310665 |
| Classification status | Preliminary recording-wise | Preliminary recording-wise | Preliminary recording-wise | Preliminary recording-wise |
| Accuracy | 0.999424 | 0.999424 | 0.984456 | 0.998849 |
| Balanced accuracy | 0.900000 | 0.900000 | 0.892494 | 0.800000 |
| Precision | 1.000000 | 1.000000 | 0.133333 | 1.000000 |
| Recall | 0.800000 | 0.800000 | 0.800000 | 0.600000 |
| F1 | 0.888889 | 0.888889 | 0.228571 | 0.750000 |
| ROC-AUC | 0.997921 | 0.827829 | 0.991455 | 0.817437 |

Patient-wise metrics: **NOT AVAILABLE** for every pipeline. All current
recordings are from the single patient CHB02, so these results must not be
interpreted as patient-wise generalization.

## 4. Dataset Statistics

| Statistic | Pipeline A | Pipeline B | Pipeline C | Pipeline D |
|---|---:|---:|---:|---:|
| Total recordings | 36 | 36 | 36 | 36 |
| Total valid windows | 63,257 | 63,257 | 63,257 | 63,257 |
| Ictal windows | 88 | 88 | 88 | 88 |
| Interictal windows | 63,169 | 63,169 | 63,169 | 63,169 |
| Ambiguous windows excluded | 0* | 0* | 0* | 0* |
| Number of patients | 1 | 1 | 1 | 1 |
| Patient-wise evaluation status | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |

`*` The feature datasets contain only valid labels 0 and 1; ambiguous windows
were excluded before dataset creation. The stored downstream summary therefore
reports zero ambiguous rows in the final datasets, not zero ambiguous windows
encountered during raw segmentation.

## 5. Interpretation

- Pipeline B preserved channel structure most closely in this Level 1 test and
  produced the largest synthetic-noise SNR improvement. It also had the highest
  silhouette score and lowest PCA dimension among the four pipelines.
- Pipeline A produced strong preliminary recording-wise classification metrics,
  but its Level 1 SNR improvement and channel correlation were below Pipeline B.
- Pipeline C retained substantial channel correlation and high preliminary
  ROC-AUC, but its negative synthetic-noise SNR improvement and slower runtime
  indicate a tradeoff requiring further investigation.
- Pipeline D improved synthetic-noise SNR and clustered better than A and D's
  PCA representation was largest, but the near-zero gamma preservation ratio is
  a major signal-preservation warning.
- No global scientific winner is declared. The current measurements are a
  single-patient, one-segment comparison and should guide further validation,
  not replace multi-patient evaluation.

## 6. Limitations

- Only patient CHB02 is available, so patient-wise generalization is not
  available with the current data.
- Level 1 uses one 60-second segment and controlled synthetic noise rather than a
  fully clean ground-truth dataset.
- Classification is preliminary recording-wise evaluation, not patient-wise
  validation.
- Runtime and memory depend on the execution environment.
- Band ratios outside 1 can reflect redistribution of relative power, not direct
  absolute power gain.
- The stored Level 1 output contains band-specific ratios; no aggregate PSD
  score is inferred here.

## 7. Reproducibility / Source Files

Verified source and result files:

- Level 1 metrics: `results/benchmarks/level1_evaluation.json`
- PCA, clustering, classification, and dataset metrics:
  `results/benchmarks/common_evaluation.json`
- Acceptance evidence: `results/final_project_verification.md`
- Common evaluation runner: `notebooks/55_common_evaluation.py`
- Level 1 runner: `notebooks/56_level1_evaluation.py`
- Cross-pipeline validator: `notebooks/57_validate_cross_pipeline_contract.py`
- Canonical full Pipeline A benchmark: `notebooks/14_benchmark_pipeline_a.py`

All four pipelines use the common Stage 0, common 4-second/50%-overlap
segmentation, common labels, and common 598-feature extractor. Generated data
and benchmark outputs remain ignored by Git.
