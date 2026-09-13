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
chebyshev = load_module("chebyshev", "pipeline/01_preprocessing/candidate_algorithms/filtering/chebyshev_type_ii_bandpass.py")
raw, _ = stage0.load_stage0("data/raw/physionet.org/chb02/chb02_01.edf", summary_file="data/raw/physionet.org/chb02/chb02-summary.txt")
raw.crop(tmin=0, tmax=60)
filtered, provenance = chebyshev.apply_chebyshev_type_ii_bandpass(raw)
assert filtered.ch_names == raw.ch_names
assert filtered.get_data().shape == raw.get_data().shape
assert np.isfinite(filtered.get_data()).all()
assert provenance["method"] == "Chebyshev Type II IIR"
print("Chebyshev Type II test passed.")
