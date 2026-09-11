import numpy as np
from sklearn.metrics import silhouette_score

def cluster_separation(Z, y):
    valid = y != -1
    if len(np.unique(y[valid])) < 2:
        return None
    return float(silhouetter_score(Z[valid], y[valid]))


def band_power_shift(X, y, n_channels = 18, n_bands = 8, W = 3):
    Xr = X.reshape(len(X), W, n_channels, n_bands)
    band_mean = Xr.mean(axis=(1,2))
    m1 = band_mean[y==1].mean(axis = 0) if (y==1).any() else np.zeros(n_bands)
    m0 = band_mean[y==0].mean(axis = 0) if (y==0).any() else np.zeros(n_bands)
    return np.arange(n_bands), m1, m0, m1-m0
    