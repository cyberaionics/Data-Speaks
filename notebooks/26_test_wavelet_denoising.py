import importlib.util
import sys

import mne
import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


wavelet = load_module(
    "wavelet_denoising",
    "pipeline/01_preprocessing/candidate_algorithms/artifact_removal/wavelet_denoising.py",
)


edf_path = "data/raw/physionet.org/chb02/chb02_01.edf"

raw = mne.io.read_raw_edf(
    edf_path,
    preload=True,
    verbose=False,
)

raw.pick("eeg")
raw.crop(tmin=0, tmax=60)

cleaned, provenance = wavelet.apply_wavelet_denoising(
    raw,
    wavelet="db4",
    level=5,
)

input_data = raw.get_data()
output_data = cleaned.get_data()

print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)
print("Sampling rate:", cleaned.info["sfreq"])
print("Channels:", len(cleaned.ch_names))

print("\nProvenance:")
for key, value in provenance.items():
    print(f"{key}: {value}")

print("\nInput RMS:", np.sqrt(np.mean(input_data ** 2)))
print("Output RMS:", np.sqrt(np.mean(output_data ** 2)))

print("Finite output:", np.isfinite(output_data).all())

assert output_data.shape == input_data.shape
assert cleaned.info["sfreq"] == raw.info["sfreq"]
assert len(cleaned.ch_names) == len(raw.ch_names)
assert np.isfinite(output_data).all()

# Ensure denoising actually changed the signal.
difference = output_data - input_data
difference_rms = np.sqrt(np.mean(difference ** 2))

print("Difference RMS:", difference_rms)

assert difference_rms > 0

print("\nWavelet denoising test passed.")
