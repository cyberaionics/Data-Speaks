import numpy as np

ZSCORE_EPS = 1e-12


def zscore_normalize_data(data, eps = ZSCORE_EPS):
    data = np.asarray(data, dtype = np.float64)
    mu = data.mean(axis = 1, keepdims = True)
    sd = data.std(axis = 1, keepdims = True)
    sd = np.where(sd<eps, eps, sd)
    normalized = (data - mu) / sd
    return normalized, {"mean":mu.flatten(), "std":sd.flatten()}


def zscore_normalize_raw(raw, eps = ZSCORE_EPS):
    out = raw.copy()
    data = out.get_data()
    normed, stats = zscore_normalize_data(data, eps)
    out._data = normed.astype(np.float64)
    return out, stats

    