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


butterworth = load_module(
    "pipeline_c_butterworth", BASE_DIR / "filtering" / "butterworth_bandpass.py"
)
asr = load_module(
    "pipeline_c_asr", BASE_DIR / "artifact_removal" / "asr_ica.py"
)
ica = load_module(
    "pipeline_c_ica", BASE_DIR / "artifact_removal" / "ica.py"
)
standardization = load_module(
    "pipeline_c_standardization", BASE_DIR / "normalization" / "standardization.py"
)
resampling = load_module(
    "pipeline_c_resampling", BASE_DIR / "resampling" / "resample_256.py"
)


@dataclass
class PipelineCConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    butterworth_order: int = 4
    asr_cutoff: float = 5.0
    asr_calibration_seconds: float = 60.0
    ica_n_components: int = 15
    ica_random_state: int = 42
    eog_channel: str = "FP1-F7"
    eog_threshold: float = 0.4
    muscle_threshold: float = 0.3
    target_sfreq: float = 256.0


def run_pipeline_c(
    raw_std: mne.io.BaseRaw,
    config: PipelineCConfig | None = None,
    recording_id: str | None = None,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Run Pipeline C: Butterworth -> ASR -> ICA -> standardization -> 256 Hz."""
    if config is None:
        config = PipelineCConfig()
    if not isinstance(raw_std, mne.io.BaseRaw):
        raise TypeError("raw_std must be an MNE Raw object.")
    if len(raw_std.ch_names) == 0:
        raise ValueError("Input contains no EEG channels.")

    provenance: dict[str, Any] = {
        "pipeline": "pipeline_C_butterworth_asr_ica_standardization_256",
        "recording_id": recording_id,
        "input_sfreq": float(raw_std.info["sfreq"]),
        "input_channels": list(raw_std.ch_names),
        "stages": [],
    }
    filtered = butterworth.apply_butterworth_bandpass(
        raw_std, config.l_freq, config.h_freq, config.butterworth_order
    )
    provenance["stages"].append({"name": "butterworth_bandpass", "l_freq": config.l_freq, "h_freq": config.h_freq, "order": config.butterworth_order})

    asr_cleaned, asr_provenance = asr.apply_asr(
        filtered, cutoff=config.asr_cutoff, calibration_seconds=config.asr_calibration_seconds
    )
    provenance["stages"].append({"name": "asr", **asr_provenance})

    ica_cleaned, ica_provenance = ica.apply_ica_artifact_removal(
        asr_cleaned,
        n_components=config.ica_n_components,
        random_state=config.ica_random_state,
        eog_ch=config.eog_channel,
        eog_threshold=config.eog_threshold,
        muscle_threshold=config.muscle_threshold,
    )
    provenance["stages"].append({"name": "ica", **ica_provenance})

    standardized, standardization_provenance = standardization.apply_per_channel_standardization(ica_cleaned)
    provenance["stages"].append({"name": "per_channel_standardization", **standardization_provenance})

    processed, resampling_provenance = resampling.resample_to_256(standardized)
    provenance["stages"].append({"name": "resampling", **resampling_provenance})

    if processed.ch_names != raw_std.ch_names:
        raise RuntimeError("Pipeline C changed channel names or order.")
    if float(processed.info["sfreq"]) != config.target_sfreq:
        raise RuntimeError("Pipeline C did not produce the target sampling rate.")
    if not np.isfinite(processed.get_data()).all():
        raise RuntimeError("Pipeline C produced non-finite data.")

    provenance["output_sfreq"] = float(processed.info["sfreq"])
    provenance["output_channels"] = list(processed.ch_names)
    provenance["output_shape"] = list(processed.get_data().shape)
    return processed, provenance
