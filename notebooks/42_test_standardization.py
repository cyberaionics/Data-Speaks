import importlib.util
import sys

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage0 = load_module("stage0", "pipeline/preprocessing_common/stage0.py")
standardization = load_module(
    "standardization",
    "pipeline/01_preprocessing/candidate_algorithms/normalization/standardization.py",
)

raw, _ = stage0.load_stage0(
    "data/raw/physionet.org/chb02/chb02_01.edf",
    summary_file="data/raw/physionet.org/chb02/chb02-summary.txt",
)
raw.crop(tmin=0, tmax=60)
standardized, provenance = standardization.apply_per_channel_standardization(raw)
data = standardized.get_data()

assert standardized.ch_names == raw.ch_names
assert standardized.info["sfreq"] == raw.info["sfreq"]
assert data.shape == raw.get_data().shape
assert np.isfinite(data).all()
assert np.allclose(np.mean(data, axis=1), 0.0, atol=1e-10)
assert np.allclose(np.std(data, axis=1), 1.0, atol=1e-10)
assert provenance["method"] == "per_channel_standardization"

print("Per-channel standardization test passed.")
