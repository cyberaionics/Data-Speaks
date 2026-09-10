"""Pipeline A — substep 3: per-channel Z-score normalization.

Rationale:
- Simple, interpretable: every channel ends with mean 0 and unit variance
  over the recording. Removes inter-channel amplitude scale differences
  (electrode impedance, amplifier gain variation) so downstream ML models
  are not dominated by high-amplitude channels.
- Chosen over robust scaling in pipeline A because conservative reference:
  after FIR + ICA the signal is comparatively clean; plain Z-score keeps
  the transform transparent and invertible from recorded statistics.

Documented decision — statistics computed over the WHOLE recording:
- Per-recording (not per-window) statistics avoid leaking future information
  into window-level normalization and keep windows comparable within a
  recording. Recording-level mean/std are stored in the provenance dict so
  the transform is exactly reproducible / invertible.
- Degenerate channels (std == 0, e.g. flatlined post-Stage-0) are left at
  zero rather than producing NaN/Inf; they should already be flagged by
  the common QC policy.
"""

import numpy as np
from mne.io import BaseRaw

EPS = 1e-12  # guards against division by zero on flat channels


def zscore_normalize(raw: BaseRaw) -> tuple[BaseRaw, dict]:
    """Per-channel Z-score. Returns (normalized copy, stats dict)."""
    data = raw.get_data()  # (n_channels, n_times), float64
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)

    safe_stds = np.where(stds < EPS, 1.0, stds)
    normalized = (data - means) / safe_stds

    raw_z = raw.copy()
    raw_z._data = normalized.astype(np.float64)

    stats = {
        "channel_means": means.ravel().tolist(),
        "channel_stds": stds.ravel().tolist(),
        "flat_channels": [
            raw.ch_names[i]
            for i in np.where(stds.ravel() < EPS)[0]
        ],
    }
    return raw_z, stats