from __future__ import annotations

import logging
import numpy as np
from scipy.special import gammaincinv, gamma

_PATCH_APPLIED = False


def _fit_eeg_distribution_patched(
    X,
    min_clean_fraction=0.25,
    max_dropout_fraction=0.1,
    fit_quantiles=None,
    step_sizes=None,
    shape_range=None,
):
    if fit_quantiles is None:
        fit_quantiles = [0.022, 0.6]
    if step_sizes is None:
        step_sizes = [0.01, 0.01]
    if shape_range is None:
        shape_range = np.arange(1.7, 3.5, 0.15)

    X = np.sort(X)
    n = len(X)

    quants = np.array(fit_quantiles)
    zbounds = []
    rescale = []
    for b in range(len(shape_range)):
        gam = gammaincinv(
            1 / shape_range[b],
            np.sign(quants - 1 / 2) * (2 * quants - 1),
        )
        zbounds.append(np.sign(quants - 1 / 2) * gam ** (1 / shape_range[b]))
        rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

    lower_min = float(np.min(quants))
    max_width = float(np.diff(quants).item())
    min_width = float(min_clean_fraction * max_width)

    cols = np.arange(
        lower_min,
        lower_min + max_dropout_fraction + step_sizes[0] * 1e-9,
        step_sizes[0],
    )
    cols = np.round(n * cols).astype(int)

    rows = np.arange(0, int(np.round(n * max_width)))

    newX = np.zeros((len(rows), len(cols)))
    for i, c in enumerate(range(len(rows))):
        newX[i] = X[c + cols]

    X1 = newX[0, :]
    newX = newX - X1

    opt_val = np.inf
    opt_lu = np.inf
    opt_bounds = np.inf
    opt_beta = np.inf

    gridsearch = np.round(
        n * np.arange(max_width, min_width, -step_sizes[1])
    )
    for m in gridsearch.astype(int):
        m_val = int(m)
        mcurr = m_val - 1
        nbins = int(np.round(3 * np.log2(1 + m_val / 2)))
        cols2 = nbins / newX[mcurr]
        H = newX[:m_val] * cols2

        hist_all = []
        for ih in range(len(cols2)):
            histcurr = np.histogram(H[:, ih], bins=np.arange(0, nbins + 1))
            hist_all.append(histcurr[0])
        hist_all = np.array(hist_all, dtype=int).T
        hist_all = np.vstack((hist_all, np.zeros(len(cols2), dtype=int)))
        logq = np.log(hist_all + 0.01)

        for k, b in enumerate(shape_range):
            bounds = zbounds[k]
            x = bounds[0] + np.arange(0.5, nbins + 0.5) / nbins * np.diff(bounds)
            p = np.exp(-np.abs(x) ** b) * rescale[k]
            p = p / np.sum(p)

            kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m_val)

            min_val = np.min(kl)
            idx = np.argmin(kl)
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = shape_range[k]
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx] + newX[m_val - 1, idx]]

    diff_bounds = float(np.diff(opt_bounds).item())
    alpha_val = float((opt_lu[1] - opt_lu[0]) / diff_bounds)
    mu_val = float(opt_lu[0] - opt_bounds[0] * alpha_val)
    beta_val = float(opt_beta)

    sig_val = float(np.sqrt((alpha_val ** 2) * gamma(3 / beta_val) / gamma(1 / beta_val)))

    return mu_val, sig_val, alpha_val, beta_val


def _block_covariance_patched(data, window=128):
    n_ch, n_times = data.shape
    offsets = np.arange(0, n_times, window)
    U = np.zeros([len(offsets), n_ch ** 2])
    data_t = data.T
    for k in range(0, window):
        idx_range = np.minimum(n_times - 1, offsets + k)
        U = U + np.reshape(
            data_t[idx_range].reshape([-1, 1, n_ch]) *
            data_t[idx_range].reshape([-1, n_ch, 1]),
            U.shape,
        )
    return np.array(U)


def apply_asrpy_numpy2_patch() -> bool:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return False

    import asrpy.asr as _asr_mod
    import asrpy.asr_utils as _asr_utils_mod

    _asr_mod.fit_eeg_distribution = _fit_eeg_distribution_patched
    _asr_mod.block_covariance = _block_covariance_patched

    _asr_utils_mod.fit_eeg_distribution = _fit_eeg_distribution_patched
    _asr_utils_mod.block_covariance = _block_covariance_patched

    _PATCH_APPLIED = True
    logging.info(
        "[pipeline_c_compat] asrpy compatibility patches applied: "
        "fit_eeg_distribution and block_covariance"
    )
    return True
