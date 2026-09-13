import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chebyshev = load_module("pipeline_d_chebyshev", BASE_DIR / "filtering" / "chebyshev_type_ii_bandpass.py")
robust_artifacts = load_module("pipeline_d_robust_artifacts", BASE_DIR / "artifact_removal" / "robust_statistical.py")
standardization = load_module("pipeline_d_standardization", BASE_DIR / "normalization" / "standardization.py")
decimation = load_module("pipeline_d_decimation", BASE_DIR / "resampling" / "decimate_256.py")


@dataclass
class PipelineDConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    chebyshev_order: int = 4
    stopband_attenuation_db: float = 40.0
    artifact_window_seconds: float = 0.5
    artifact_z_threshold: float = 6.0
    target_sfreq: float = 256.0


def run_pipeline_d(
    raw_std: mne.io.BaseRaw,
    config: PipelineDConfig | None = None,
    recording_id: str | None = None,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Run Pipeline D: Chebyshev II -> robust detection -> standardization -> decimation."""
    if config is None:
        config = PipelineDConfig()
    if not isinstance(raw_std, mne.io.BaseRaw):
        raise TypeError("raw_std must be an MNE Raw object.")
    if len(raw_std.ch_names) == 0:
        raise ValueError("Input contains no EEG channels.")

    provenance: dict[str, Any] = {
        "pipeline": "pipeline_D_chebyshev_robust_standardization_decimation_256",
        "recording_id": recording_id,
        "input_sfreq": float(raw_std.info["sfreq"]),
        "input_channels": list(raw_std.ch_names),
        "stages": [],
    }
    filtered, filter_provenance = chebyshev.apply_chebyshev_type_ii_bandpass(
        raw_std,
        config.l_freq,
        config.h_freq,
        config.chebyshev_order,
        config.stopband_attenuation_db,
    )
    provenance["stages"].append({"name": "chebyshev_type_ii_bandpass", **filter_provenance})
    cleaned, artifact_provenance = robust_artifacts.apply_robust_statistical_artifact_detection(
        filtered, config.artifact_window_seconds, config.artifact_z_threshold
    )
    provenance["stages"].append({"name": "robust_statistical_artifact_detection", **artifact_provenance})
    standardized, standardization_provenance = standardization.apply_per_channel_standardization(cleaned)
    provenance["stages"].append({"name": "per_channel_standardization", **standardization_provenance})
    processed, decimation_provenance = decimation.decimate_to_256(standardized)
    provenance["stages"].append({"name": "decimation", **decimation_provenance})

    if processed.ch_names != raw_std.ch_names:
        raise RuntimeError("Pipeline D changed channel names or order.")
    if float(processed.info["sfreq"]) != config.target_sfreq:
        raise RuntimeError("Pipeline D did not produce the target sampling rate.")
    if not np.isfinite(processed.get_data()).all():
        raise RuntimeError("Pipeline D produced non-finite data.")
    provenance["output_sfreq"] = float(processed.info["sfreq"])
    provenance["output_channels"] = list(processed.ch_names)
    provenance["output_shape"] = list(processed.get_data().shape)
    return processed, provenance
