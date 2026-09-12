import numpy as np


def hjorth_features(epoch):
    out = np.empty((epoch.shape[0], 2), dtype=np.float32)
    for i, ch in enumerate(epoch):
        d1 = np.diff(ch)
        d2 = np.diff(d1)
        v0 = np.var(ch)
        v1 = np.var(d1) if len(d1) > 1 else 0.0
        v2 = np.var(d2) if len(d2) > 1 else 0.0
        mob = np.sqrt(v1 / v0) if v0 > 0 else 0.0
        comp = np.sqrt(v2 / v1) if v1 > 0 else 0.0
        out[i] = (mob, comp)
    return out


def hjorth_for_stacked(stacked):
    n, W, c, _ = stacked.shape
    X = np.empty((n, W * c * 2), dtype = np.float32)
    for i in range(n):
        chunks = [hjorth_features(stacked[i, w]).flatten() for w in range(W)]
        X[i] = np.concatenate(chunks)
    return X