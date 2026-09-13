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


robust = load_module(
    "robust_scaling",
    "pipeline/01_preprocessing/candidate_algorithms/normalization/robust_scaling.py",
)


edf_path = "data/raw/physionet.org/chb02/chb02_01.edf"

raw = mne.io.read_raw_edf(
    edf_path,
    preload=True,
    verbose=False,
)

raw.pick("eeg")
raw.crop(tmin=0, tmax=60)

scaled, provenance = robust.apply_robust_scaling(raw)

data = scaled.get_data()

medians = np.median(data, axis=1)
iqrs = (
    np.percentile(data, 75, axis=1)
    - np.percentile(data, 25, axis=1)
)

print("Input shape:", raw.get_data().shape)
print("Output shape:", data.shape)
print("Sampling rate:", scaled.info["sfreq"])
print("Channels:", len(scaled.ch_names))

print("\nMaximum absolute median:")
print(np.max(np.abs(medians)))

print("\nIQR range:")
print("Minimum:", np.min(iqrs))
print("Maximum:", np.max(iqrs))

print("\nFinite output:", np.isfinite(data).all())

print("\nProvenance:")
for key, value in provenance.items():
    print(f"{key}: {value}")

assert data.shape == raw.get_data().shape
assert np.isfinite(data).all()

# Median should be approximately zero.
assert np.max(np.abs(medians)) < 1e-10

# IQR should be approximately one.
assert np.allclose(
    iqrs,
    1.0,
    atol=1e-10,
)

print("\nRobust scaling test passed.")
