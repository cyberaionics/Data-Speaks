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
pipeline_c = load_module("pipeline_c", "pipeline/01_preprocessing/candidate_algorithms/pipeline_c.py")

edf_path = "data/raw/physionet.org/chb02/chb02_16.edf"
summary_path = "data/raw/physionet.org/chb02/chb02-summary.txt"

raw, _ = stage0.load_stage0(edf_path, summary_file=summary_path)
raw.crop(tmin=0, tmax=180)
processed, provenance = pipeline_c.run_pipeline_c(raw, recording_id="chb02_16_first180s")

data = processed.get_data()
means = np.mean(data, axis=1)
stds = np.std(data, axis=1)

assert processed.ch_names == raw.ch_names
assert processed.info["sfreq"] == 256.0
assert data.shape == raw.get_data().shape
assert np.isfinite(data).all()
assert np.allclose(means, 0.0, atol=1e-10)
assert np.allclose(stds, 1.0, atol=1e-10)
assert [stage["name"] for stage in provenance["stages"]] == [
    "butterworth_bandpass", "asr", "ica", "per_channel_standardization", "resampling"
]

print("Pipeline C end-to-end test passed.")
print("ASR retained calibration fraction:", provenance["stages"][1]["calibration_retained_fraction"])
print("ICA excluded components:", provenance["stages"][2]["excluded_components"])
