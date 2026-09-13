import importlib.util
import sys

import mne
import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)

    # Register module before executing it.
    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


butterworth = load_module(
    "butterworth",
    "pipeline/01_preprocessing/candidate_algorithms/filtering/butterworth_bandpass.py",
)


edf_path = (
    "data/raw/physionet.org/chb02/chb02_01.edf"
)

raw = mne.io.read_raw_edf(
    edf_path,
    preload=True,
    verbose=False,
)

# Keep EEG channels only, matching Stage 0 behavior.
raw.pick("eeg")

# Test only the first 60 seconds.
raw.crop(tmin=0, tmax=60)

filtered = butterworth.apply_butterworth_bandpass(
    raw,
    l_freq=1.0,
    h_freq=40.0,
    order=4,
)

print("Input shape:", raw.get_data().shape)
print("Output shape:", filtered.get_data().shape)
print("Sampling rate:", filtered.info["sfreq"])
print("Channels:", len(filtered.ch_names))

data = filtered.get_data()

print("Finite values:", np.isfinite(data).all())
print(
    "RMS:",
    np.sqrt(np.mean(data ** 2)),
)

assert filtered.info["sfreq"] == raw.info["sfreq"]
assert filtered.get_data().shape == raw.get_data().shape
assert len(filtered.ch_names) == len(raw.ch_names)
assert np.isfinite(data).all()

print("\nButterworth band-pass test passed.")
