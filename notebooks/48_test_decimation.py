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


decimation = load_module("decimation", "pipeline/01_preprocessing/candidate_algorithms/resampling/decimate_256.py")
rng = np.random.default_rng(42)
raw = mne.io.RawArray(rng.normal(size=(2, 2048)), mne.create_info(["EEG1", "EEG2"], 512.0, "eeg"), verbose=False)
output, provenance = decimation.decimate_to_256(raw)
assert output.ch_names == raw.ch_names
assert output.info["sfreq"] == 256.0
assert output.get_data().shape == (2, 1024)
assert np.isfinite(output.get_data()).all()
assert provenance["decimation_factor"] == 2
print("Decimation test passed.")
