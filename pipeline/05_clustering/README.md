# Stage 05: Unsupervised Clustering

Stage 05 investigates whether unsupervised grouping algorithms (such as K-Means and DBSCAN) operating on the reduced PCA representation of continuous EEG can naturally separate seizure dynamics from interictal background EEG **without using ground-truth labels**.

---

## 1. Algorithmic Formulations

### K-Means Clustering ($k = 2 \dots 6$)
Minimizes within-cluster sum of squares (WCSS / Inertia):
$$J = \sum_{j=1}^k \sum_{x \in S_j} \|x - \mu_j\|^2$$
where $\mu_j$ is the centroid of cluster $S_j$.
Evaluated for:
- $k=2$ (testing the binary hypothesis: interictal vs. ictal)
- $k \in \{2, 3, 4, 5, 6\}$ (testing multi-state background dynamics)

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Groups points with dense neighborhoods ($N_\epsilon(p) \ge \text{MinPts}$) and marks sparse points as outliers / noise (label $-1$).
- Tested with $\epsilon = 12.0$ and $\text{MinPts} = 15$ on the full cohort PCA space to evaluate if rare ictal events present as isolated density outliers.

---

## 2. Quantitative Evaluation Metrics

All clusterings were executed strictly unsupervised. Post-hoc comparisons against true clinical labels were measured using:
1. **Silhouette Score**: Intrinsic geometric separation of clusters:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
2. **Adjusted Rand Index (ARI)**: Similarity between cluster partition and true labels, adjusted for chance ($-1$ to $1$, where $0$ is random).
3. **Normalized Mutual Information (NMI)**: Shared entropy between predicted clusters and true states ($0$ to $1$).

---

## 3. Empirical Results

### Full Subject CHB01 Cohort ($72,951$ Windows, 110-PC Space)

Data source: [`results/tables/chb01_clustering_metrics.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_clustering_metrics.csv)

| Algorithm | Clusters ($K$) | Silhouette Score | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
|---|:---:|:---:|:---:|:---:|
| **K-Means ($k=2$)** | 2 | **0.1527** | 0.0034 | 0.0040 |
| **K-Means ($k=3$)** | 3 | 0.1438 | 0.0027 | 0.0035 |
| **K-Means ($k=4$)** | 4 | 0.1335 | 0.0026 | 0.0057 |
| **K-Means ($k=5$)** | 5 | 0.1376 | 0.0034 | 0.0056 |
| **K-Means ($k=6$)** | 6 | 0.1109 | 0.0027 | 0.0056 |
| **DBSCAN ($\epsilon=12.0, \text{MinPts}=15$)** | 4 | 0.1003 | **0.0061** | **0.0102** |

### Single Recording Benchmark (`chb01_03`, $1,799$ Windows, 100-PC Space)

Data source: [`results/tables/chb01_03_clustering_metrics.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_03_clustering_metrics.csv)

| Algorithm | Clusters ($K$) | Silhouette Score | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
|---|:---:|:---:|:---:|:---:|
| **K-Means ($k=2$)** | 2 | 0.1830 | -0.0010 | 0.0038 |
| K-Means ($k=3$) | 3 | 0.1867 | 0.0131 | 0.0264 |
| K-Means ($k=4$) | 4 | **0.1983** | 0.0221 | 0.0688 |
| K-Means ($k=5$) | 5 | 0.1747 | 0.0159 | 0.0556 |
| K-Means ($k=6$) | 6 | 0.1357 | 0.0098 | 0.0467 |
| **DBSCAN ($\epsilon=10, \text{MinPts}=8$)** | 1 (+ noise) | 0.0070 | -0.0100 | 0.0018 |

---

## 4. Key Scientific Insights

1. **Unsupervised K-Means does NOT separate seizures from background EEG**:
   - For $k=2$, the cluster partitions group background physiological states (e.g. sleep/wake transitions, baseline amplitude shifts) rather than class labels.
   - Seizure windows are split across clusters in proportions reflecting background density.
   - Adjusted Rand Index ($\text{ARI} = 0.0034$) and $\text{NMI} = 0.0040$ are near zero.
2. **Why this occurs**:
   - **Extreme class imbalance ($0.30\%$)**: In continuous recordings, seizures represent only 0.30% of windows. K-Means variance minimization is dominated by continuous state variations in the 99.70% interictal background.
   - **Hyper-spherical assumption**: K-Means assumes isotropic, balanced clusters. A tiny 226-sample positive class out of 72,951 windows cannot form a standalone Voronoi partition in 110-dimensional space.
3. **Conclusion for the Pipeline**:
   - Empirically demonstrates that **unsupervised clustering alone is insufficient for automated seizure detection** and rigorously justifies the necessity of supervised classification (Stage 06) or dedicated anomaly detection algorithms.

---

## 5. Generated Artifacts

- **Cluster Metrics Summary Tables**:
  - `results/tables/chb01_clustering_metrics.csv` (72,951 cohort)
  - `results/tables/chb01_03_clustering_metrics.csv` (1,799 benchmark)
- **Window-level Cluster Assignments**:
  - `results/tables/chb01_clustered_windows.csv`
- **Figures** (in [`results/figures/clustering/`](file:///C:/Users/Kavya/Data-Speaks/results/figures/clustering/)):
  - `kmeans_k2_clusters.png`: K-Means 2-cluster partition side-by-side with ground truth.
  - `kmeans_elbow_silhouette.png`: WCSS elbow curve and silhouette score profile for $k=2..6$.
  - `dbscan_clusters.png`: Density-based outlier detection projection.

---

## 6. Usage & Execution

```powershell
# Run standalone clustering pass on existing PCA matrix
C:\Users\Kavya\anaconda3\python.exe pipeline/run_chb01_complete_batch.py --pass3-only
```
