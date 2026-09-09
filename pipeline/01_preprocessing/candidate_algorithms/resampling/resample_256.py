from __future__ import annotations

from typing import Any

import mne


TARGET_SFREQ = 256.0


def resample_to_256(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """
    Ensure the EEG output uses the common 256 Hz sampling rate.

    If the input is already 256 Hz, return an unchanged copy.
    Otherwise, resample using MNE's polyphase FIR anti-aliasing.
    """

    input_sfreq = float(raw.info["sfreq"])

    if input_sfreq <= 0:
        raise ValueError(
            f"Invalid input sampling rate: {input_sfreq}"
        )

    output = raw.copy()

    if input_sfreq != TARGET_SFREQ:
        output.resample(
            TARGET_SFREQ,
            method="polyphase",
            verbose=False,
        )

        operation = "resampled"
    else:
        operation = "no_op_already_256_hz"

    provenance: dict[str, Any] = {
        "input_sampling_rate_hz": input_sfreq,
        "output_sampling_rate_hz": float(
            output.info["sfreq"]
        ),
        "target_sampling_rate_hz": TARGET_SFREQ,
        "operation": operation,
        "anti_aliasing": "MNE polyphase FIR",
    }

    return output, provenance
