# Stage 04: Dimensionality Reduction (PCA)

Stage 04 implements Principal Component Analysis (PCA) on the multichannel EEG feature matrix extracted in Stage 03, with strict feature/metadata separation, leakage-safe standardization, empirical cumulative variance tracking, and projection.

---

## 1. Mathematical Formulation & Architecture

Given an input feature matrix $X \in \mathbb{R}^{N \times D}$ where $N$ is the number of 4-second windows (e.g. $N = 1,799$ for `chb01_03`, $N = 72,951$ for the full `chb01` cohort) and $D = 442$ numerical EEG features (from 17 bipolar channels):

1. **Isolation of Metadata & Labels**:
   - The 7 metadata columns (`patient_id`, `recording_id`, `window_id`, `start_sec`, `end_sec`, `duration_sec`, `label`) are strictly detached before mathematical processing.
   - **Labels are never used** in fitting `StandardScaler` or `PCA`.
2. **Feature Standardization**:
   - Every feature is zero-mean unit-variance scaled:
     $$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
3. **Principal Component Decomposition**:
   - The empirical covariance matrix $\Sigma = \frac{1}{N-1}\tilde{X}^T \tilde{X} \in \mathbb{R}^{D \times D}$ is decomposed into orthogonal eigenvectors $W = [w_1, w_2, \dots, w_D]$ and corresponding eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$:
     $$\Sigma w_k = \lambda_k w_k$$
   - The explained variance ratio of component $k$ is:
     $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j}$$
   - Coordinates in principal component space:
     $$Z = \tilde{X} W_k \in \mathbb{R}^{N \times k}$$

---

## 2. Empirical Cumulative Explained Variance Analysis

### Full Subject CHB01 Cohort ($N = 72,951$ Windows $\times 442$ Features)

| Threshold | Principal Components Required | Feature Space Reduction |
|---|:---:|:---:|
| **80% Explained Variance** | **35 PCs** | **92.1% reduction** |
| **90% Explained Variance** | **73 PCs** | **83.5% reduction** |
| **95% Explained Variance (Primary)** | **110 PCs** | **75.1% reduction** |
| **99% Explained Variance** | **217 PCs** | **50.9% reduction** |

### Single Recording Benchmark (`chb01_03`, $N = 1,799$ Windows)

| Threshold | Principal Components Required | Feature Space Reduction |
|---|:---:|:---:|
| **80% Explained Variance** | **33 PCs** | **92.5% reduction** |
| **90% Explained Variance** | **66 PCs** | **85.1% reduction** |
| **95% Explained Variance** | **100 PCs** | **77.4% reduction** |
| **99% Explained Variance** | **195 PCs** | **55.9% reduction** |

---

## 3. Exploratory Visualization & Scientific Interpretation

Figures generated in [`results/figures/pca/`](file:///C:/Users/Kavya/Data-Speaks/results/figures/pca/):

1. **PC1 vs PC2 Scatter Plot**:
   - Projects windows into the 2D subspace defined by PC1 and PC2.
   - Points are color-coded by class: Interictal (`0`) vs Ictal (`1`).
   - **Scientific finding**: Seizure windows form trajectories within specific regions of PC1/PC2 but remain embedded within the high-density interictal distribution. Unsupervised PCA maximizes total signal variance (dominated by background state fluctuations), rather than discriminative class margin.
2. **Scree Plot & Cumulative Variance Curve**:
   - Demonstrates rapid eigenvalue decay across the top components, followed by an extended tail.
3. **Feature Loadings**:
   - **PC1**: Dominated by broad-band amplitude metrics across frontal and central channels.
   - **PC2**: Strongly loaded by beta/gamma band powers and line length contrasted against negative relative delta power.

---

## 4. Generated Artifacts

- **PCA Variance Decomposition Summaries**:
  - `results/tables/chb01_pca_variance.csv` (72,951 cohort)
  - `results/tables/chb01_03_pca_variance.csv` (1,799 benchmark)
- **Feature Loading Matrices**:
  - `results/tables/chb01_03_pca_loadings.csv`
- **Visualizations**:
  - `results/figures/pca/pca_scree_plot.png`
  - `results/figures/pca/pca_scatter_pc1_pc2.png`

---

## 5. Usage & Execution

From the repository root:

### Run unit tests

```powershell
python pipeline/04_dimensionality_reduction/test_dimensionality_reduction.py
```
