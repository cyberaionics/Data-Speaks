from __future__ import annotations

from typing import Any

import mne
import numpy as np
from scipy.signal import decimate


TARGET_SFREQ = 256.0


def decimate_to_256(raw: mne.io.BaseRaw) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Reduce an integer-multiple sampling rate to 256 Hz with zero-phase FIR decimation."""
    input_sfreq = float(raw.info["sfreq"])
    if input_sfreq < TARGET_SFREQ:
        raise ValueError("Pipeline D cannot decimate an input below 256 Hz.")
    ratio = input_sfreq / TARGET_SFREQ
    factor = int(round(ratio))
    if not np.isclose(ratio, factor):
        raise ValueError(
            f"Pipeline D requires an integer decimation ratio to 256 Hz; got {ratio}."
        )
    if factor == 1:
        return raw.copy(), {
            "method": "FIR zero-phase decimation",
            "input_sampling_rate_hz": input_sfreq,
            "output_sampling_rate_hz": input_sfreq,
            "target_sampling_rate_hz": TARGET_SFREQ,
            "decimation_factor": 1,
            "operation": "no_op_already_256_hz",
        }

    data = decimate(raw.get_data(), factor, axis=1, ftype="fir", zero_phase=True)
    info = raw.info.copy()
    with info._unlock():
        info["sfreq"] = TARGET_SFREQ
    output = mne.io.RawArray(data, info, verbose=False)
    return output, {
        "method": "FIR zero-phase decimation",
        "input_sampling_rate_hz": input_sfreq,
        "output_sampling_rate_hz": TARGET_SFREQ,
        "target_sampling_rate_hz": TARGET_SFREQ,
        "decimation_factor": factor,
        "operation": "decimated",
    }
