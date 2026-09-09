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
pipeline_d = load_module("pipeline_d", "pipeline/01_preprocessing/candidate_algorithms/pipeline_d.py")
raw, _ = stage0.load_stage0("data/raw/physionet.org/chb02/chb02_16.edf", summary_file="data/raw/physionet.org/chb02/chb02-summary.txt")
raw.crop(tmin=0, tmax=180)
processed, provenance = pipeline_d.run_pipeline_d(raw, recording_id="chb02_16_first180s")
data = processed.get_data()
assert processed.ch_names == raw.ch_names
assert processed.info["sfreq"] == 256.0
assert data.shape == raw.get_data().shape
assert np.isfinite(data).all()
assert np.allclose(np.mean(data, axis=1), 0.0, atol=1e-10)
assert np.allclose(np.std(data, axis=1), 1.0, atol=1e-10)
assert [stage["name"] for stage in provenance["stages"]] == ["chebyshev_type_ii_bandpass", "robust_statistical_artifact_detection", "per_channel_standardization", "decimation"]
print("Pipeline D end-to-end test passed.")
