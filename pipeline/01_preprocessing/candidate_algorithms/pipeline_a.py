from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.util

import mne


@dataclass
class PipelineAConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0

    ica_n_components: int = 15
    ica_random_state: int = 42

    eog_channel: str = "FP1-F7"
    eog_threshold: float = 0.4
    muscle_threshold: float = 0.3

    target_sfreq: float = 256.0


def _load_function(
    file_path: Path,
    function_name: str,
):
    """Load a function from a module path."""

    spec = importlib.util.spec_from_file_location(
        function_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, function_name)


def run_pipeline_a(
    raw_std: mne.io.BaseRaw,
    config: PipelineAConfig | None = None,
    recording_id: str = "unknown",
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """
    Run Pipeline A on Stage-0 standardized EEG.

    Pipeline:

        FIR 1-40 Hz
            ↓
        ICA artifact removal
            ↓
        per-channel Z-score
            ↓
        256 Hz

    Parameters
    ----------
    raw_std:
        Standardized EEG produced by Common Stage 0.
    config:
        Pipeline A configuration.
    recording_id:
        Identifier used in provenance.

    Returns
    -------
    processed:
        Final processed EEG.
    provenance:
        Dictionary describing every preprocessing step.
    """

    if config is None:
        config = PipelineAConfig()

    if not isinstance(raw_std, mne.io.BaseRaw):
        raise TypeError(
            "raw_std must be an MNE Raw object."
        )

    provenance: dict[str, Any] = {
        "pipeline": "pipeline_A_fir_ica_zscore_256",
        "recording_id": recording_id,
        "input": {
            "sampling_rate_hz": float(
                raw_std.info["sfreq"]
            ),
            "n_channels": len(raw_std.ch_names),
            "channel_names": list(raw_std.ch_names),
            "n_samples": raw_std.n_times,
        },
    }

    # ---------------------------------------------------------
    # Step 1: FIR band-pass
    # ---------------------------------------------------------

    fir_path = (
        Path(__file__).resolve().parent
        / "filtering"
        / "fir_bandpass.py"
    )

    apply_fir_bandpass = _load_function(
        fir_path,
        "apply_fir_bandpass",
    )

    filtered = apply_fir_bandpass(
        raw_std,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
    )

    provenance["fir"] = {
        "method": "firwin",
        "phase": "zero",
        "low_hz": config.l_freq,
        "high_hz": config.h_freq,
    }

    # ---------------------------------------------------------
    # Step 2: ICA artifact removal
    # ---------------------------------------------------------

    ica_path = (
        Path(__file__).resolve().parent
        / "artifact_removal"
        / "ica.py"
    )

    apply_ica_artifact_removal = _load_function(
        ica_path,
        "apply_ica_artifact_removal",
    )

    cleaned, ica_provenance = (
        apply_ica_artifact_removal(
            filtered,
            n_components=config.ica_n_components,
            random_state=config.ica_random_state,
            eog_ch=config.eog_channel,
            eog_threshold=config.eog_threshold,
            muscle_threshold=config.muscle_threshold,
        )
    )

    provenance["ica"] = ica_provenance

    # ---------------------------------------------------------
    # Step 3: Per-channel recording-level Z-score
    # ---------------------------------------------------------

    zscore_path = (
        Path(__file__).resolve().parent
        / "normalization"
        / "zscore.py"
    )

    apply_zscore = _load_function(
        zscore_path,
        "apply_zscore",
    )

    normalized, zscore_provenance = apply_zscore(
        cleaned
    )

    provenance["normalization"] = (
        zscore_provenance
    )

    # ---------------------------------------------------------
    # Step 4: Ensure 256 Hz
    # ---------------------------------------------------------

    resample_path = (
        Path(__file__).resolve().parent
        / "resampling"
        / "resample_256.py"
    )

    resample_to_256 = _load_function(
        resample_path,
        "resample_to_256",
    )

    processed, resample_provenance = (
        resample_to_256(normalized)
    )

    provenance["resampling"] = (
        resample_provenance
    )

    # ---------------------------------------------------------
    # Final output checks
    # ---------------------------------------------------------

    if float(processed.info["sfreq"]) != config.target_sfreq:
        raise RuntimeError(
            "Pipeline A did not produce the required "
            f"{config.target_sfreq} Hz output."
        )

    if len(processed.ch_names) != len(
        raw_std.ch_names
    ):
        raise RuntimeError(
            "Pipeline A changed the number of channels."
        )

    provenance["output"] = {
        "sampling_rate_hz": float(
            processed.info["sfreq"]
        ),
        "n_channels": len(
            processed.ch_names
        ),
        "n_samples": processed.n_times,
        "shape": [
            len(processed.ch_names),
            processed.n_times,
        ],
        "format": "MNE FIF-compatible Raw",
    }

    return processed, provenance
