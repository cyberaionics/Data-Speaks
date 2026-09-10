# Stage 05: Unsupervised Clustering

Stage 05 investigates whether unsupervised grouping algorithms (such as K-Means and DBSCAN) operating on the 100-dimensional PCA representation of continuous EEG (`chb01_03_pca_reduced.csv`, 95% cumulative variance) can naturally separate seizure dynamics from interictal background EEG **without using ground-truth labels**.

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
- Tested with $\epsilon = 10.0$ and $\text{MinPts} = 8$ to evaluate if rare ictal events (21 windows out of 1,799) present as isolated anomalous clusters or density outliers.

---

## 2. Quantitative Evaluation Metrics

All clusterings were executed strictly unsupervised. Post-hoc comparisons against true clinical labels were measured using:
1. **Silhouette Score**: Intrinsic geometric separation of clusters:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
2. **Adjusted Rand Index (ARI)**: Similarity between cluster partition and true labels, adjusted for chance ($-1$ to $1$, where $0$ is random).
3. **Normalized Mutual Information (NMI)**: Shared entropy between predicted clusters and true states ($0$ to $1$).

---

## 3. Results on Real Data (`chb01_03`)

| Algorithm | Clusters | Silhouette Score | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
|---|:---:|:---:|:---:|:---:|
| **K-Means ($k=2$)** | 2 | 0.1830 | **-0.0010** | **0.0038** |
| **K-Means ($k=3$)** | 3 | 0.1867 | 0.0131 | 0.0264 |
| **K-Means ($k=4$)** | 4 | **0.1983** | 0.0221 | 0.0688 |
| **K-Means ($k=5$)** | 5 | 0.1747 | 0.0159 | 0.0556 |
| **K-Means ($k=6$)** | 6 | 0.1357 | 0.0098 | 0.0467 |
| **DBSCAN ($\epsilon=10, \text{MinPts}=8$)** | 1 (+ noise) | 0.0070 | -0.0100 | 0.0018 |

### Cluster vs. True Label Contingency ($k=2$)
```
True Label:     Normal (0)    Seizure (1)
Cluster 0:         851             5
Cluster 1:         927            16
```

---

## 4. Key Scientific Insights for Midterm Report

1. **Unsupervised K-Means does NOT separate seizures from background EEG**:
   - For $k=2$, the cluster partition roughly bisects the 1,778 normal windows (851 in Cluster 0 vs 927 in Cluster 1).
   - Seizure windows (21 total) are split across both clusters (5 in Cluster 0, 16 in Cluster 1).
   - The Adjusted Rand Index ($\text{ARI} = -0.0010$) and $\text{NMI} = 0.0038$ are approximately zero.
2. **Why this occurs (The Scientific Story)**:
   - **Extreme class imbalance ($< 1\%$)**: In 1 hour of recording, seizures represent only 1.17% of windows (21 out of 1,799). K-Means variance minimization is overwhelmed by the continuous state variations in the background EEG (sleep stages, blinking, postural shifts).
   - **Hyper-spherical assumption**: K-Means assumes isotropic, balanced clusters. A tiny 21-sample cluster cannot drive a global Voronoi partition in 100-dimensional space.
3. **Conclusion for the Pipeline**:
   - This empirically demonstrates that **unsupervised clustering alone is insufficient for automated seizure detection** and rigorously justifies the necessity of supervised discrimination (Stage 06 Classification) or dedicated anomaly detection.

---

## 5. Generated Artifacts

- **Cluster Metrics Summary Table**:
  [`results/tables/chb01_03_clustering_metrics.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_03_clustering_metrics.csv)
- **Window-level Cluster Assignments**:
  [`results/tables/chb01_03_clustered_windows.csv`](file:///C:/Users/Kavya/Data-Speaks/results/tables/chb01_03_clustered_windows.csv)
- **Figures** (in [`results/figures/clustering/`](file:///C:/Users/Kavya/Data-Speaks/results/figures/clustering/)):
  - `kmeans_k2_clusters.png`: K-Means 2-cluster partition side-by-side with ground truth.
  - `kmeans_elbow_silhouette.png`: WCSS elbow curve and silhouette score profile for $k=2..6$.
  - `dbscan_clusters.png`: Density-based outlier detection projection.
