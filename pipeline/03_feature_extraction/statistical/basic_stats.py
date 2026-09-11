import numpy as np
from scipy import stats

def basic_stats(epoch):
    out = np.empty((epoch.shape[0], 6), dtype=np.float32)
    for i, ch in enumerate(epoch):
        out[i]=(float(np.mean(ch)), float(np.std(ch)), float(stats.skew(ch)), float(stats.kurtosis(ch)), float(np.sqrt(np.mean(ch**2))), np.ptp(ch))
    return out


def stats_for_stacked(stacked):
    n, W, c, _ = stacked.shape
    X = np.empty((n, W * c * 6), dtype = np.float32)
    for i in range(n):
        chunks = [basic_stats(stacked[i,w]).flatten() for w in range(W)]
        X[i] = np.concatenate(chunks)
    return X