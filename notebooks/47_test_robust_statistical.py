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
robust = load_module("robust", "pipeline/01_preprocessing/candidate_algorithms/artifact_removal/robust_statistical.py")
raw, _ = stage0.load_stage0("data/raw/physionet.org/chb02/chb02_01.edf", summary_file="data/raw/physionet.org/chb02/chb02-summary.txt")
raw.crop(tmin=0, tmax=60)
data = raw.get_data()
data[0, 1000] = 1.0
raw._data[:] = data
cleaned, provenance = robust.apply_robust_statistical_artifact_detection(raw)
assert cleaned.ch_names == raw.ch_names
assert cleaned.get_data().shape == data.shape
assert np.isfinite(cleaned.get_data()).all()
assert provenance["artifact_samples_total"] > 0
assert cleaned.get_data()[0, 1000] != 1.0
print("Robust statistical artifact detection test passed.")
