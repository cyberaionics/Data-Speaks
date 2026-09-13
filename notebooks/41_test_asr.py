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
butterworth = load_module(
    "butterworth", "pipeline/01_preprocessing/candidate_algorithms/filtering/butterworth_bandpass.py"
)
asr = load_module(
    "asr_ica", "pipeline/01_preprocessing/candidate_algorithms/artifact_removal/asr_ica.py"
)
ica = load_module(
    "ica", "pipeline/01_preprocessing/candidate_algorithms/artifact_removal/ica.py"
)

raw, _ = stage0.load_stage0(
    "data/raw/physionet.org/chb02/chb02_16.edf",
    summary_file="data/raw/physionet.org/chb02/chb02-summary.txt",
)
raw.crop(tmin=0, tmax=180)
filtered = butterworth.apply_butterworth_bandpass(raw)
cleaned, provenance = asr.apply_asr(filtered)
ica_cleaned, ica_provenance = ica.apply_ica_artifact_removal(
    cleaned,
    eog_ch="FP1-F7",
)

input_data = filtered.get_data()
output_data = cleaned.get_data()
ica_output_data = ica_cleaned.get_data()

assert cleaned.ch_names == filtered.ch_names
assert cleaned.info["sfreq"] == filtered.info["sfreq"]
assert output_data.shape == input_data.shape
assert np.isfinite(output_data).all()
assert ica_cleaned.ch_names == cleaned.ch_names
assert ica_output_data.shape == output_data.shape
assert np.isfinite(ica_output_data).all()
assert 0 < provenance["calibration_retained_fraction"] <= 1
assert np.any(np.abs(output_data - input_data) > 0)

print("ASR + ICA component test passed.")
print("Calibration retained fraction:", provenance["calibration_retained_fraction"])
print("ICA excluded components:", ica_provenance["excluded_components"])
