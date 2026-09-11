import numpy as np


def channel_covariance_eig(data, sfreq, t0, t1):
    i0, i1 = int(t0 * sfreq), int(t1 * sfreq)
    seg = data[:, i0:i1]
    seg = seg - seg.mean(axis=1, keepdims=True)
    C = np.cov(seg)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]