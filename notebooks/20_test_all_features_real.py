import importlib.util
import numpy as np
import mne


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feature_module = load_module(
    "feature_extractor",
    "pipeline/03_feature_extraction/feature_extractor.py",
)


# Load Pipeline A output
path = "data/processed/pipeline_A/chb02_16_raw.fif"

raw = mne.io.read_raw_fif(
    path,
    preload=True,
    verbose=False,
)

sfreq = raw.info["sfreq"]


def test_window(start_sec):
    start_sample = int(start_sec * sfreq)
    stop_sample = start_sample + int(4 * sfreq)

    window = raw.get_data(
        start=start_sample,
        stop=stop_sample,
    )

    print(f"\nWindow: {start_sec:.1f}–{start_sec + 4:.1f} s")
    print("Shape:", window.shape)

    features = feature_module.extract_features(
        window,
        sfreq,
    )

    print("Feature count:", len(features))

    assert window.shape == (23, 1024)
    assert len(features) == 598

    assert all(
        np.isfinite(value)
        for value in features.values()
    )

    print("All 598 features are finite.")

    print("Example features:")
    for name in [
        "ch1_mean",
        "ch1_rms",
        "ch1_alpha_power",
        "ch1_beta_relative_power",
        "ch1_median",
        "ch1_skewness",
    ]:
        print(f"  {name}: {features[name]:.6f}")


print("Sampling rate:", sfreq)
print("Channels:", len(raw.ch_names))

# Interictal window: well before seizure
test_window(100.0)

# Ictal window: inside the 130–212 s seizure
test_window(150.0)

print("\nReal EEG all-feature extraction test passed.")
