import importlib.util
import numpy as np
import mne


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load feature extractor
features_module = load_module(
    "time_features",
    "pipeline/03_feature_extraction/time_domain/features.py",
)

# Load one processed Pipeline A recording
path = "data/processed/pipeline_A/chb02_16_raw.fif"
raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

sfreq = raw.info["sfreq"]

# First 4-second window
window = raw.get_data(start=0, stop=int(4 * sfreq))

print("Window shape:", window.shape)
print("Sampling rate:", sfreq)
print("Channels:", len(raw.ch_names))

features = features_module.extract_time_domain_features(window)

print("\nFeature shapes:")
for name, values in features.items():
    print(f"{name}: {values.shape}")

flat = features_module.flatten_time_domain_features(window)

print("\nTotal flattened features:", len(flat))

print("\nFirst 10 features:")
for i, (name, value) in enumerate(flat.items()):
    if i >= 10:
        break
    print(f"{name}: {value:.6f}")

# Basic validation
assert window.shape == (len(raw.ch_names), int(4 * sfreq))
assert len(features) == 8
assert all(values.shape == (len(raw.ch_names),) for values in features.values())
assert len(flat) == len(raw.ch_names) * 8
assert all(np.isfinite(value) for value in flat.values())

print("\nReal EEG feature extraction test passed.")
