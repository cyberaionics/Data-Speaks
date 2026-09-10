# Data Speaks!! — Epileptic Seizure Pattern Mining & EEG Analysis
> **Course Project:** Data Speaks!! (Educational Data Mining & Data Science)  
> **Institution:** Indian Institute of Technology Dharwad (IIT Dharwad)  
> **Marks Distribution:** 30 Marks (20 Midterm + 10 Endterm)  
> **Dataset:** [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (Boston Children's Hospital / MIT)

---

## 📌 Executive Summary & Objective

Epilepsy is one of the most prevalent neurological conditions globally, characterized by sudden, recurrent neuro-electrical disturbances (seizures). Scalp Electroencephalography (EEG) records continuous multi-channel electrical brain activity at microvolt resolution. 

As part of the **"Data Speaks!!"** course initiative, our team is analyzing the benchmark **CHB-MIT Scalp EEG Dataset**. The core objectives are:
1. **Explore and clean** multi-channel clinical pediatric EEG time-series data.
2. **Benchmark 4 distinct preprocessing pipelines** head-to-head to isolate noise, remove artifacts, and preserve true seizure morphology.
3. **Extract predictive features** and train Machine Learning models to detect epileptic seizures automatically under extreme class imbalance.
4. **Deliver an interactive web dashboard** hosted via GitHub showcasing our findings, visualizations, and live model predictions.

---

## 👥 Team Pipeline Division (4 Approaches)

To systematically study how preprocessing affects downstream machine learning, our team divided into 4 specialized signal-processing approaches with a shared channel, windowing, and evaluation policy:

| Approach | Assigned Pipeline | Key Characteristics | Lead |
| :--- | :--- | :--- | :--- |
| **Approach 1** | **FIR Bandpass $\to$ ICA Artifact Removal $\to$ Z-Score Normalization** | Linear phase, zero phase distortion, classical matrix decomposition for ocular/muscle noise. | Teammate 1 |
| **Approach 2** | **Butterworth Bandpass $\to$ Wavelet Denoising $\to$ Robust Scaling $\to$ Polyphase Resampling** | Smooth IIR filtering, time-frequency multi-resolution wavelet thresholding, outlier-resistant scaling. | Manish |
| **Approach 3** | **Butterworth Bandpass $\to$ ASR + ICA $\to$ Channel Standardization $\to$ Resampling** | Modern automated Artifact Subspace Reconstruction (ASR) with principal component subspace rejection. | Kavyanjali |
| **Approach 4** | **Chebyshev Type II Bandpass $\to$ Robust Statistical Artifact Detection $\to$ Decimation** | Ultra-sharp stopband rolloff, zero passband distortion, blazingly fast statistical outlier checks, 60.9% dimensional compression. | **Avni (This Module)** |

---

## 🏛️ Dataset Overview & Harmonization Standard

- **Subjects:** 24 pediatric patients (`chb01` – `chb24`) with intractable epilepsy.
- **Sampling Frequency:** $256 \text{ Hz}$ ($256$ samples per second per channel).
- **Format:** European Data Format (`.edf`) binary time-series paired with clinical annotations (`chbXX-summary.txt`).
- **Standard 18-Channel Bipolar Policy (International 10–20 Double Banana Montage):**
  - *Left Temporal:* `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`
  - *Left Parasagittal:* `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`
  - *Right Parasagittal:* `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`
  - *Right Temporal:* `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2`
  - *Midline:* `FZ-CZ`, `CZ-PZ`
- **Windowing (Segmentation):** $4.0\text{-second}$ sliding windows with $50\%$ overlap ($2.0\text{-second}$ step).
- **Labeling Standard:** Window labeled `1` (Ictal / Seizure) if $\ge 50\%$ duration intersects documented clinical seizure times; otherwise `0` (Interictal / Normal).

---

## Project Architecture

```
Data-Speaks/
├── data/
│   ├── raw/physionet.org/          ← CHB-MIT (gitignored)
│   └── processed/
├── pipeline/
│   ├── 01_preprocessing/
│   │   ├── README.md               ← common preprocessing contract
│   │   ├── baseline/
│   │   ├── candidate_algorithms/
│   │   │   ├── filtering/
│   │   │   ├── artifact_removal/
│   │   │   ├── normalization/
│   │   │   └── resampling/
│   │   └── evaluation/             ← shared evaluator
│   ├── 02_segmentation/            ← README + candidates + evaluation
│   ├── 03_feature_extraction/      ← time/frequency/statistical + evaluation
│   ├── 04_dimensionality_reduction/← pca / svd / ica + evaluation
│   ├── 05_clustering/              ← kmeans / hierarchical / dbscan + evaluation
│   └── 06_classification/          ← LR / RF / SVM / KNN + evaluation
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_comparison.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_dimensionality_reduction.ipynb
│   ├── 05_clustering_comparison.ipynb
│   └── 06_classification_comparison.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── benchmarks/
├── reports/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 🚀 Completed Work: Phase 1 (Midterm Milestone — 20 Marks)

For the midterm evaluation, our work focused on **data ingestion, schema auditing, preprocessing verification on `chb01`, and Exploratory Data Analysis (EDA)**.

### 1. Approach 4 Implementation & Pipeline Architecture
- **Chebyshev Type II Filter ($0.5 - 45\text{ Hz}$):**
  - Designed an order-4 IIR filter with stopband attenuation of $30\text{ dB}$ using `scipy.signal.cheby2`.
  - Filtered bidirectionally with `sosfiltfilt` (zero phase delay, zero brain-wave distortion).
  - Power Spectral Density (PSD) confirms $>30\text{ dB}$ suppression of the $60\text{ Hz}$ US electrical powerline interference.
- **Decimation ($256\text{ Hz} \to 128\text{ Hz}$):**
  - Implemented 2x downsampling with automatic lowpass anti-aliasing via `scipy.signal.decimate`.
  - Preserved the full Nyquist band up to $64\text{ Hz}$ while reducing disk storage and memory by **$60.9\%$**.
- **Robust Statistical Artifact Detection:**
  - Evaluates each 4-second epoch for physiological validity:
    1. *Flatline Check:* Variance $< 1.0\ \mu V^2$ (loose or fallen electrode).
    2. *Lead Pop / Step Jump:* Instantaneous single-sample differential $> 200\ \mu V$ (static shock/cable bump).
    3. *Abnormal Channel Variance:* Channel variance $> 15\times$ median channel variance.
    4. *Extreme Saturation:* Amplitude $> 800\ \mu V$.
  - Preserves genuine clinical seizures (which naturally exhibit high-voltage rhythmic activity between $200\text{--}900\ \mu V$) while safely flagging true hardware/muscle noise.

### 2. Experimental Verification on Subject `chb01`

| Recording ID | Raw Size | Processed Size | Clean Epochs | Flagged Artifacts (%) | Seizure Epochs | Total Runtime | Throughput vs Real-Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`chb01_03.edf`** | 40.44 MB | **15.82 MB** | 1,788 | 11 (0.61%) | **21 epochs** | **0.899 s** | **4,002x** |
| **`chb01_01.edf`** | 40.44 MB | **15.82 MB** | 1,424 | 375 (20.84%) | 0 (Baseline) | **0.927 s** | **3,883x** |
| **`chb01_04.edf`** | 40.44 MB | **15.82 MB** | 1,754 | 45 (2.50%) | **14 epochs** | **1.657 s** | **2,172x** |

### 3. Generated Exploratory Data Analysis (EDA)
- **Spectral PSD Comparison:** Clear visual proof of line hum removal and flat passband response (`plots/psd_chebyshev_comparison.png`).
- **Time-Domain Waveform Comparison:** Direct side-by-side visualization of calm interictal background vs. synchronous spike-and-wave discharges during seizure (`plots/seizure_vs_normal_eeg.png`).
- **Artifact Rejection Distribution:** Quantified breakdown of artifact categories across recordings (`plots/artifact_detection_breakdown.png`).
- **Jupyter Notebook Deliverable:** Self-contained, reproducible notebook [Approach4_CHB01_Preprocessing_EDA.ipynb](file:///C:/Users/avani/.gemini/antigravity/scratch/chbmit_pipeline_approach4/Approach4_CHB01_Preprocessing_EDA.ipynb).

---

## 🔮 Future Roadmap: Phase 2 (Endterm Milestone — 10 Marks)

In accordance with the **"Data Speaks!!"** project guidelines, here is the complete plan for the final submission:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA SPEAKS!! ENDTERM EXECUTION ROADMAP                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
    ┌────────────────────────┬─────────────┴────────────┬────────────────────────┐
    ▼                        ▼                          ▼                        ▼
[1. Dataset Scaling]   [2. Feature Eng.]      [3. Machine Learning]    [4. Deliverables]
  • Scale to chb02-24    • 5 Brain Wave Bands   • Supervised Classifiers  • GitHub Web Page
  • Cross-subject pool   • Statistical Moments  • Unsupervised Clustering • 3-5 Page Report
  • Unified metadata     • Cross-Channel Corr   • Dimensionality Red.     • Model Deployment
```

### Step 1: Dataset-Wide Scaling (`chb01` $\to$ `chb24`)
- Run the validated Approach 4 pipeline across all remaining recordings with clinical seizures.
- Compile a single unified dataset manifest linking patient demographic data (Age, Gender) with seizure onset latency.

### Step 2: Comprehensive Feature Engineering
Extract domain-grounded feature vectors for every 4-second epoch across 3 families:
1. **Frequency-Domain (Spectral Bands via Welch's PSD):**
   - Delta ($\delta$: $0.5 - 4\text{ Hz}$): Slow-wave seizure discharge power.
   - Theta ($\theta$: $4 - 8\text{ Hz}$): Temporal lobe rhythmic activity.
   - Alpha ($\alpha$: $8 - 13\text{ Hz}$): Posterior background rhythm.
   - Beta ($\beta$: $13 - 30\text{ Hz}$): Fast background rhythm.
   - Gamma ($\gamma$: $30 - 45\text{ Hz}$): High-frequency bursts.
   - Spectral Ratios: $(\delta + \theta) / (\alpha + \beta)$ and Spectral Entropy.
2. **Time-Domain (Statistical & Morphological):**
   - Energy / Variance ($\sigma^2$), Peak-to-Peak amplitude (PTP).
   - Skewness and Kurtosis (capturing heavy-tailed spike distributions).
   - Zero-Crossing Rate (ZCR) and Hjorth Parameters (Activity, Mobility, Complexity).
3. **Spatial & Connectivity Features:**
   - Inter-channel Pearson cross-correlation matrix (capturing hemispheric hyper-synchronization).

### Step 3: Machine Learning & Modeling (Aligned with Syllabus Rubric)
- **Supervised Classification (Seizure Detection):**
  - Baseline: Logistic Regression & Support Vector Classifier (RBF Kernel).
  - Ensembles: **Random Forest** & **XGBoost / LightGBM** (evaluating feature importance rankings).
  - Deep Learning (Optional / Advanced): 1D-CNN or CNN-LSTM operating directly on multi-channel time slices.
- **Unsupervised Clustering (Brain State Discovery):**
  - Apply **K-Means** and **DBSCAN** on extracted feature embeddings to discover whether unannotated sub-clinical seizure patterns or distinct sleep stages emerge naturally.
- **Dimensionality Reduction:**
  - **PCA (Principal Component Analysis):** Linear projection to 2D/3D showing separability of ictal vs. interictal manifolds.
  - **t-SNE / UMAP:** Non-linear manifold learning for visual validation of class separation.

### Step 4: Tackling Imbalance & Validation Protocol
- **Imbalance Mitigation:** Since seizures represent $<0.5\%$ of data, use **SMOTE (Synthetic Minority Over-sampling)**, random undersampling of interictal background, and `class_weight='balanced'`.
- **Validation Schemes:**
  - *Patient-Specific:* 5-Fold Stratified Cross-Validation within a single patient.
  - *Cross-Patient Generalization:* Leave-One-Subject-Out (LOSO) cross-validation to evaluate real-world clinical applicability.
- **Metrics:** Prioritize **Sensitivity (Recall)**, **Specificity**, **Precision**, **F1-Score**, and **AUROC / AUPRC** rather than deceptive overall accuracy.

### Step 5: "Data Speaks!!" Interactive Web Page Deliverable
As mandated in the assignment specifications, we will develop and host a responsive web application:
- **Hosting:** GitHub Pages / Streamlit Community Cloud.
- **Interactive Features:**
  - *Live Dataset Explorer:* Select patient and recording to inspect clinical metadata.
  - *Interactive Waveform Viewer:* Scrub through 18-channel EEG signals before, during, and after seizures.
  - *Audio / Spectral Sonification:* Visualize live frequency power shifts during seizure onset.
  - *Pipeline Comparison Dashboard:* Interactive side-by-side comparison of all 4 team approaches.
  - *Live Prediction Playground:* Upload a sample EEG window to test our trained ML classifier.

### Step 6: Final 3–5 Page Project Report
- Structure according to standard IEEE/ACM scientific paper formatting:
  1. *Abstract & Introduction:* Clinical background and problem formulation.
  2. *Dataset & Preprocessing:* Comparative review of the 4 pipelines and justification for Approach 4.
  3. *Feature Engineering & Modeling:* Methodological description of classifiers and clustering.
  4. *Results & Discussion:* Performance tables, confusion matrices, ROC curves, and clinical interpretations.
  5. *Conclusion & Future Work:* Real-time hardware deployment feasibility on wearable EEG devices.

---

## 📁 Repository Structure

```
chbmit_pipeline_approach4/
│
├── README.md                                  <- Comprehensive project guide & roadmap
├── approach4_pipeline.py                      <- Modular preprocessing library (Chebyshev II, Decimation, Artifacts)
├── run_pipeline_chb01.py                      <- Execution and benchmarking script on chb01
├── make_notebook.py                           <- Automated generator for assignment notebook
├── Approach4_CHB01_Preprocessing_EDA.ipynb    <- Primary Jupyter Notebook deliverable for midterm
│
├── output/
│   └── approach4_chb01_metrics.csv            <- Benchmark metrics table for chb01 recordings
│
└── plots/
    ├── psd_chebyshev_comparison.png           <- Power Spectral Density (PSD) showing 60 Hz hum removal
    ├── seizure_vs_normal_eeg.png              <- Waveform comparison: Interictal vs Ictal seizure discharge
    └── artifact_detection_breakdown.png       <- Distribution of flagged non-physiological artifacts
```

---

## 💻 How to Run the Project

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Core scientific libraries:
  ```bash
  pip install numpy scipy pandas matplotlib scikit-learn mne
  ```

### Running the Preprocessing Pipeline
To execute the complete pipeline on `chb01` and regenerate all performance metrics:
```bash
python run_pipeline_chb01.py
```

### Running the Jupyter Notebook
Launch JupyterLab or VS Code to interactively step through the cells and view visualizations:
```bash
jupyter lab Approach4_CHB01_Preprocessing_EDA.ipynb
```

---

## 📖 References
1. **Goldberger, A. L., et al.** (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23), e215-e220.
2. **Shoeb, A. H.** (2010). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, Massachusetts Institute of Technology.
3. **Gramfort, A., et al.** (2013). *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience, 7, 267.
