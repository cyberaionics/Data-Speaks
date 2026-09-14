# CHB‑MIT Cohort Pipeline (Approach 4)

## Overview
This repository implements **Approach 4** for preprocessing, feature extraction, and cohort‑level seizure detection on the CHB‑MIT Scalp EEG Database. The pipeline is fully reproducible, runs on a single machine, and produces a Random Forest classifier evaluated with patient‑level leave‑one‑group‑out cross‑validation.

---

## Repository Structure
```
README.md                <-- this file (artifact)
context.md               <-- architectural decisions and rationale
cohort_pipeline.py       <-- main driver (preprocessing, CV, model training)
approach4_pipeline.py    <-- channel handling & feature extraction utilities
make_notebook.py         <-- assembles a Jupyter notebook with results
tests/
│   test_cv_grouping.py  <-- verifies patient‑wise CV correctness
processed_dataset/       <-- generated NumPy arrays and CSV metadata (see below)
    X_cohort.npy
    y_cohort.npy
    patient_ids.npy
    recording_ids.npy
    metadata_cohort.csv
    seizure_events_metadata.csv
output/
    cohort_model_evaluation_metrics.csv
    <various>_eda_plot.png
    Approach4_CHB01_Preprocessing_EDA.ipynb
requirements.txt          <-- Python dependencies (use `pip install -r requirements.txt`)
```

**Note:** `processed_dataset/` and `output/` are *generated* by the pipeline and are **not** tracked in version control. Add them to `.gitignore`.

---

## Installation
1. **Python 3.12+** (or newer) is required.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` contains `mne`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, and other lightweight packages.
3. Ensure the raw CHB‑MIT EDF files are placed under a directory of your choice (e.g., `data/raw/`); the pipeline expects the path to be configurable inside `cohort_pipeline.py`.

---

## How to Use the Pipeline
```powershell
# From the repository root
python cohort_pipeline.py
```
The script performs the following steps:
1. **Load & clean** all EDF recordings (non‑EEG channels are removed, montage is fixed).
2. **Windowing:** 4 s windows with 2 s step (50 % overlap) are created and stacked into 6‑s classification windows.
3. **Feature extraction:** spectral power in eight bands (0.5–25 Hz) for each of the 7‑band × 18‑channel layout → 126 features per window.
4. **Cross‑validation:** `LeaveOneGroupOut` on `patient_id` eliminates any patient‑level leakage.
5. **Model training:** a `RandomForestClassifier` (100 trees, max depth 12, class‑weight `balanced`).
6. **Evaluation:** computes event‑level sensitivity, latency, false‑alarm rate (per hour & per 24 h), AUROC, and AUPRC.
7. **Reporting:** writes a CSV summary to `output/cohort_model_evaluation_metrics.csv`, saves EDA plots, and creates a Jupyter notebook (`Approach4_CHB01_Preprocessing_EDA.ipynb`).

### Selecting Subsets of Patients
If you only want to process a subset, edit the `SELECTED_PATIENT_IDS` variable near the top of `cohort_pipeline.py` (or add a CLI argument) like:
```python
SELECTED_PATIENT_IDS = {"chb01", "chb03"}  # None processes all patients
```
The pipeline will filter the dataset before CV, requiring no further code changes.

### Running Multiple Targets
The script already processes **all selected recordings** in one run. To experiment with different configurations (e.g., feature sets, hyper‑parameters), simply invoke the script multiple times with the desired changes; each run writes its own output files.

---

## Metrics Summary (from the latest run)
| Metric | Value |
|--------|-------|
| **Cohort Subjects Evaluated** | 18 |
| **Total Recordings Processed** | 52 |
| **Total Evaluated Hours** | 76.16 |
| **Total Clinical Seizures** | 72 |
| **Event‑Level Sensitivity (%)** | 61.11 |
| **Epoch‑Level Sensitivity (%)** | 42.67 |
| **Epoch‑Level Precision (%)** | 59.51 |
| **Mean Detection Latency (s)** | 12.48 |
| **False Alarms per 24 Hours** | 206.72 |
| **False Alarms per Hour** | 8.61 |
| **AUROC** | 0.9088 |
| **AUPRC** | 0.5142 |

*The Random Forest model (100 trees, max depth 12) achieves the above performance when evaluated with patient‑level leave‑one‑group‑out CV.*

---

## Limitations
- **Patient‑specific evaluation:** The current protocol holds out recordings **within the same patient**, so the model does **not** generalize across patients. A true patient‑independent assessment would require a different CV scheme.
- **Channel variability:** Some recordings (e.g., `chb01`) have duplicate or missing channels, resulting in 17 retained channels instead of the nominal 18.
- **Feature simplicity:** Only raw FFT power sums are used. Normalized band power, log‑transforms, or time‑domain features could improve discrimination.
- **Event‑level metrics** are computed per recording; seizure‑level sensitivity (multiple seizures per EDF) is not yet implemented.
- **Computation cost:** The Random Forest is modest, but more complex models (e.g., RBF‑SVM) become expensive for the full cohort.

---

## Future Work & End‑Term Planning
The mid‑term deliverable covered data cleaning, preprocessing, and exploratory analysis. The end‑term goals expand the project to a full classification and inference system:
1. **Patient‑independent model** – implement `LeaveOnePatientOut` or stratified group CV across patients to assess true generalization.
2. **Enhanced feature set** – add Hjorth parameters, entropy, and normalized band power; explore deep‑learning embeddings.
3. **Threshold optimization** – replace the fixed 0.5 decision threshold with ROC‑based or cost‑sensitive tuning.
4. **Seizure‑level evaluation** – compute sensitivity per seizure rather than per recording, and report precision‑recall curves.
5. **Real‑time inference pipeline** – design a streaming mode that processes incoming EDF streams and outputs detections with low latency.
6. **Scientific analysis** – perform statistical tests (e.g., paired t‑tests) to compare models, report confidence intervals, and discuss clinical relevance.
7. **Documentation & reproducibility** – package the pipeline as a pip‑installable module, provide Dockerfiles, and publish the processed dataset as a Zenodo snapshot for reproducibility.

---

## References
1. **Goldberger, A. L., et al.** *PhysioNet: A repository of biomedical data.* 2010. (CHB‑MIT dataset source)
2. **MNE‑Python Development Team.** *MNE‑Python: Open source toolbox for EEG/MEG data analysis.* 2023.
3. **Pedregosa, F., et al.** *Scikit‑learn: Machine learning in Python.* JMLR, 12, 2825‑2830, 2011.
4. Relevant conference papers on EEG seizure detection (e.g., *ICASSP 2022 – Deep learning for seizure prediction*). *(Add specific citations as needed.)*

---

*This README was prepared to facilitate reproducible research and collaborative development.*
