import numpy as np
from mne.io import BaseRaw

EPS = 1e-12


def zscore_normalize(raw: BaseRaw) -> tuple[BaseRaw, dict]:
    data = raw.get_data()
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)

    safe_stds = np.where(stds < EPS, 1.0, stds)
    normalized = (data - means) / safe_stds

    raw_z = raw.copy()
    raw_z._data = normalized.astype(np.float64)

    stats = {"channel_means": means.ravel().tolist(), "channel_stds": stds.ravel().tolist(),"flat_channels": [raw.ch_names[i] for i in np.where(stds.ravel() < EPS)[0]]}
    return raw_z, stats