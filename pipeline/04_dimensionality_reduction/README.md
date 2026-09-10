# Stage 04: Dimensionality Reduction (PCA)

Stage 04 implements Principal Component Analysis (PCA) on the multichannel EEG features extracted in Stage 03 (`chb01_03_features.csv`), with strict feature/metadata separation, leakage-safe standardization, empirical cumulative variance tracking, and projection.

---

## 1. Mathematical Formulation & Architecture

Given an input feature matrix $X \in \mathbb{R}^{N \times D}$ where $N = 1,799$ 4-second windows and $D = 442$ numerical EEG features (from 17 bipolar channels):

1. **Isolation of Metadata & Labels**:
   - The 7 metadata columns (`patient_id`, `recording_id`, `window_id`, `start_sec`, `end_sec`, `duration_sec`, `label`) are strictly detached before mathematical processing.
   - **Labels are never used** in fitting `StandardScaler` or `PCA`.
2. **Feature Standardization**:
   - Because features span diverse units (variance $\sim 1.0$, band powers $\sim 10^{-2}$, line length $\sim 50$), every feature is standardized:
     $$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
     where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $j$.
3. **Principal Component Decomposition**:
   - The empirical covariance matrix $\Sigma = \frac{1}{N-1}\tilde{X}^T \tilde{X} \in \mathbb{R}^{D \times D}$ is decomposed into orthogonal eigenvectors $W = [w_1, w_2, \dots, w_D]$ and corresponding eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$:
     $$\Sigma w_k = \lambda_k w_k$$
   - The explained variance ratio of component $k$ is:
     $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j}$$
   - The coordinates in principal component space are obtained via linear projection:
     $$Z = \tilde{X} W_k \in \mathbb{R}^{N \times k}$$

---

## 2. Empirical Cumulative Explained Variance Analysis (`chb01_03`)

PCA was fitted over the full spectrum of $D = 442$ components on `chb01_03` to determine empirical variance thresholds without heuristic guessing:

| Metric | Value | Interpretation / Reduction |
|---|:---:|---|
| **PC1 Explained Variance** | **23.23%** | Captures overall amplitude/energy dynamics |
| **PC2 Explained Variance** | **16.11%** | Captures higher-frequency power (beta/gamma) vs Hjorth complexity |
| **Top 2 PCs Combined** | **39.34%** | 2D subspace representation |
| **Top 5 PCs Combined** | **55.31%** | More than half the total multichannel feature variance |
| **Top 10 PCs Combined** | **64.14%** | Captures nearly two-thirds of total variance |
| **80% Variance Threshold** | **33 PCs** | **92.5% dimensionality reduction** (from 442 to 33 features) |
| **90% Variance Threshold** | **66 PCs** | **85.1% dimensionality reduction** (from 442 to 66 features) |
| **95% Variance Threshold** | **100 PCs** | **77.4% dimensionality reduction** (from 442 to 100 features) |
| **99% Variance Threshold** | **195 PCs** | **55.9% dimensionality reduction** (from 442 to 195 features) |

*Note: The primary reduced dataset output preserves the **95% cumulative variance threshold** ($k = 100$ components).*

---

## 3. Exploratory Visualization & Scientific Interpretation

All figures are generated in [`results/figures/dimensionality_reduction/`](file:///C:/Users/Kavya/Data-Speaks/results/figures/dimensionality_reduction/):

1. **PC1 vs PC2 Scatter Plot (`pca_scatter_pc1_pc2.png`)**:
   - Projects all 1,799 windows into the 2D subspace defined by PC1 (23.23%) and PC2 (16.11%).
   - Points are color-coded by class: **Interictal (`0`, n=1778)** in blue, **Ictal (`1`, n=21)** in crimson triangles.
   - **Scientific finding**: The 21 ictal windows form a cluster along specific trajectories of PC1/PC2 but remain embedded within the high-density interictal distribution. **PCA is an unsupervised variance-maximization technique, not a class-separability tool.** Because interictal background dynamics constitute 98.8% of the recording, PCA maximizes total signal variance (dominated by background state fluctuations), rather than discriminative margin.
2. **Scree Plot & Cumulative Variance Curve (`pca_explained_variance_scree.png`)**:
   - Shows the rapid eigenvalue decay across the first 15–20 components, followed by an extended tail.
3. **Feature Loadings (`pca_top_loadings_pc1_pc2.png`)**:
   - **PC1**: Dominated by broad-band amplitude metrics across frontal and central channels (positive loadings: `F4-C4_rms`, `F4-C4_std`, `FP1-F7_rms`, `F3-C3_rms`; negative loadings: channel minima).
   - **PC2**: Strongly loaded by beta/gamma band powers and line length (positive: `FP1-F7_beta_power`, `F8-T8_line_length`, `C3-P3_gamma_power`) contrasted against negative Hjorth complexity and relative delta power.

---

## 4. Generated Artifacts

- **Reduced Feature Table (95% Variance)**:
  [`results/tables/chb01_03_pca_reduced.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_03_pca_reduced.csv)
  - Shape: `(1799, 107)` — 7 metadata columns + 100 principal components.
  - Zero NaNs, Zero Infs.
- **PCA Variance Decomposition Summary**:
  [`results/tables/chb01_03_pca_variance.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_03_pca_variance.csv)
- **Feature Loadings Matrix**:
  [`results/tables/chb01_03_pca_loadings.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_03_pca_loadings.csv)
- **Visualizations**:
  - `results/figures/dimensionality_reduction/pca_scatter_pc1_pc2.png`
  - `results/figures/dimensionality_reduction/pca_explained_variance_scree.png`
  - `results/figures/dimensionality_reduction/pca_top_loadings_pc1_pc2.png`

---

## 5. Usage & Execution

```powershell
# Run unit tests
C:\Users\Kavya\anaconda3\python.exe C:\Users\Kavya\Data-Speaks\pipeline\04_dimensionality_reduction\test_dimensionality_reduction.py

# Run real data execution on chb01_03
C:\Users\Kavya\anaconda3\python.exe C:\Users\Kavya\Data-Speaks\pipeline\04_dimensionality_reduction\run_dimensionality_reduction_chb01_03.py
```
