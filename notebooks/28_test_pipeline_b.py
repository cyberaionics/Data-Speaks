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


pipeline_b = load_module(
    "pipeline_b",
    "pipeline/01_preprocessing/candidate_algorithms/pipeline_b.py",
)


edf_path = "data/raw/physionet.org/chb02/chb02_01.edf"

raw = mne.io.read_raw_edf(
    edf_path,
    preload=True,
    verbose=False,
)

# Keep EEG channels only, matching the Stage 0 standardized input
# for this CHB02 recording.
raw.pick("eeg")

# Use first 60 seconds for the end-to-end test.
raw.crop(tmin=0, tmax=60)

config = pipeline_b.PipelineBConfig()

processed, provenance = pipeline_b.run_pipeline_b(
    raw,
    config=config,
    recording_id="chb02_01",
)

data = processed.get_data()

print("Input shape:", raw.get_data().shape)
print("Output shape:", data.shape)

print("Input sampling rate:", raw.info["sfreq"])
print("Output sampling rate:", processed.info["sfreq"])

print("Input channels:", len(raw.ch_names))
print("Output channels:", len(processed.ch_names))

print("\nFinal data:")
print("Finite:", np.isfinite(data).all())
print("Minimum:", np.min(data))
print("Maximum:", np.max(data))
print("Mean:", np.mean(data))
print("Std:", np.std(data))

print("\nPipeline stages:")
for stage in provenance["stages"]:
    print(stage)

assert len(processed.ch_names) == len(raw.ch_names)
assert processed.ch_names == raw.ch_names
assert processed.info["sfreq"] == 256.0
assert data.shape == raw.get_data().shape
assert np.isfinite(data).all()

print("\nPipeline B end-to-end test passed.")
