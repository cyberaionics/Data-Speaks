import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import mne


BASE_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


butterworth = load_module(
    "pipeline_b_butterworth",
    BASE_DIR / "filtering" / "butterworth_bandpass.py",
)

wavelet = load_module(
    "pipeline_b_wavelet",
    BASE_DIR / "artifact_removal" / "wavelet_denoising.py",
)

robust = load_module(
    "pipeline_b_robust",
    BASE_DIR / "normalization" / "robust_scaling.py",
)

resampling = load_module(
    "pipeline_b_resampling",
    BASE_DIR / "resampling" / "resample_256.py",
)


@dataclass
class PipelineBConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    butterworth_order: int = 4

    wavelet: str = "db4"
    wavelet_level: int = 5

    target_sfreq: float = 256.0


def run_pipeline_b(
    raw_std: mne.io.BaseRaw,
    config: PipelineBConfig | None = None,
    recording_id: str | None = None,
):
    """
    Run Pipeline B:

    Butterworth band-pass
        ->
    Wavelet denoising
        ->
    Robust scaling
        ->
    256 Hz resampling
    """

    if config is None:
        config = PipelineBConfig()

    if len(raw_std.ch_names) == 0:
        raise ValueError("Input contains no EEG channels")

    provenance = {
        "pipeline": "pipeline_B_butterworth_wavelet_robust_256",
        "recording_id": recording_id,
        "input_sfreq": float(raw_std.info["sfreq"]),
        "input_channels": list(raw_std.ch_names),
        "stages": [],
    }

    # -----------------------------------------------------
    # Stage 1: Butterworth band-pass
    # -----------------------------------------------------

    filtered = butterworth.apply_butterworth_bandpass(
        raw_std,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        order=config.butterworth_order,
    )

    provenance["stages"].append(
        {
            "name": "butterworth_bandpass",
            "l_freq": config.l_freq,
            "h_freq": config.h_freq,
            "order": config.butterworth_order,
        }
    )

    # -----------------------------------------------------
    # Stage 2: Wavelet denoising
    # -----------------------------------------------------

    denoised, wavelet_provenance = (
        wavelet.apply_wavelet_denoising(
            filtered,
            wavelet=config.wavelet,
            level=config.wavelet_level,
        )
    )

    provenance["stages"].append(
        {
            "name": "wavelet_denoising",
            **wavelet_provenance,
        }
    )

    # -----------------------------------------------------
    # Stage 3: Robust scaling
    # -----------------------------------------------------

    scaled, robust_provenance = (
        robust.apply_robust_scaling(denoised)
    )

    provenance["stages"].append(
        {
            "name": "robust_scaling",
            **robust_provenance,
        }
    )

    # -----------------------------------------------------
    # Stage 4: Resample to 256 Hz
    # -----------------------------------------------------

    processed, resampling_provenance = (
        resampling.resample_to_256(
            scaled,
        )
    )

    provenance["stages"].append(
        {
            "name": "resampling",
            **resampling_provenance,
        }
    )

    # -----------------------------------------------------
    # Final checks
    # -----------------------------------------------------

    if len(processed.ch_names) != len(raw_std.ch_names):
        raise RuntimeError(
            "Pipeline B changed the number of channels"
        )

    if processed.info["sfreq"] != config.target_sfreq:
        raise RuntimeError(
            "Pipeline B did not produce the target sampling rate"
        )

    if not processed.ch_names == raw_std.ch_names:
        raise RuntimeError(
            "Pipeline B changed the channel order"
        )

    if not processed.get_data().dtype:
        raise RuntimeError(
            "Pipeline B produced invalid data"
        )

    provenance["output_sfreq"] = float(
        processed.info["sfreq"]
    )
    provenance["output_channels"] = list(
        processed.ch_names
    )
    provenance["output_shape"] = list(
        processed.get_data().shape
    )

    return processed, provenance
