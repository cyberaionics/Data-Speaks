import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).parent

_SUBPATHS = [
    "01_preprocessing/baseline",
    "01_preprocessing/candidate_algorithms",
    "01_preprocessing/candidate_algorithms/filtering",
    "01_preprocessing/candidate_algorithms/artifact_removal",
    "01_preprocessing/candidate_algorithms/normalization",
    "01_preprocessing/candidate_algorithms/resampling",
    "01_preprocessing/evaluation",

    "02_segmentation/candidate_algorithms",
    "02_segmentation/evaluation",

    "03_feature_extraction/time_domain",
    "03_feature_extraction/frequency_domain",
    "03_feature_extraction/statistical",
    "03_feature_extraction/evaluation",

    "04_dimensionality_reduction/pca",
    "04_dimensionality_reduction/svd",
    "04_dimensionality_reduction/ica",
    "04_dimensionality_reduction/evaluation",

    "05_clustering/kmeans",
    "05_clustering/hierarchical",
    "05_clustering/dbscan",
    "05_clustering/evaluation",

    "06_classification/logistic_regression",
    "06_classification/random_forest",
    "06_classification/svm",
    "06_classification/knn",
    "06_classification/evaluation",
]

for _sub in _SUBPATHS:
    _p = _PIPELINE_ROOT / _sub
    if _p.exists():
        _s = str(_p)
        if _s not in sys.path():
            sys.path.insert(0, _s)