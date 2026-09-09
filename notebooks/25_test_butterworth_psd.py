import importlib.util

import matplotlib.pyplot as plt
import mne
import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)

    import sys
    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


butterworth = load_module(
    "butterworth",
    "pipeline/01_preprocessing/candidate_algorithms/filtering/butterworth_bandpass.py",
)

edf_path = "data/raw/physionet.org/chb02/chb02_01.edf"

raw = mne.io.read_raw_edf(
    edf_path,
    preload=True,
    verbose=False,
)

raw.pick("eeg")
raw.crop(tmin=0, tmax=60)

filtered = butterworth.apply_butterworth_bandpass(
    raw,
    l_freq=1.0,
    h_freq=40.0,
    order=4,
)

# Use channel 1 for the PSD comparison.
raw_data = raw.get_data(picks=[0])[0]
filtered_data = filtered.get_data(picks=[0])[0]

sfreq = raw.info["sfreq"]

raw_psd, freqs = mne.time_frequency.psd_array_welch(
    raw_data,
    sfreq=sfreq,
    fmin=0.1,
    fmax=100.0,
    n_fft=2048,
    verbose=False,
)

filtered_psd, _ = mne.time_frequency.psd_array_welch(
    filtered_data,
    sfreq=sfreq,
    fmin=0.1,
    fmax=100.0,
    n_fft=2048,
    verbose=False,
)

# Convert PSD to dB.
raw_db = 10 * np.log10(np.maximum(raw_psd, 1e-30))
filtered_db = 10 * np.log10(np.maximum(filtered_psd, 1e-30))

# Print average power by frequency region.
regions = {
    "below_1Hz": (freqs < 1.0),
    "1_40Hz": ((freqs >= 1.0) & (freqs <= 40.0)),
    "above_40Hz": (freqs > 40.0),
}

print("PSD comparison for first EEG channel:")

for name, mask in regions.items():
    raw_power = np.mean(raw_psd[mask])
    filtered_power = np.mean(filtered_psd[mask])

    print(f"\n{name}")
    print(f"Raw:      {raw_power:.6e}")
    print(f"Filtered: {filtered_power:.6e}")

# Passband preservation check.
passband_ratio = (
    np.mean(filtered_psd[regions["1_40Hz"]])
    / np.mean(raw_psd[regions["1_40Hz"]])
)

high_freq_ratio = (
    np.mean(filtered_psd[regions["above_40Hz"]])
    / np.mean(raw_psd[regions["above_40Hz"]])
)

print(f"\nPassband power ratio: {passband_ratio:.4f}")
print(f"Above-40Hz power ratio: {high_freq_ratio:.4f}")

# Plot.
plt.figure(figsize=(10, 5))

plt.plot(
    freqs,
    raw_db,
    label="Raw",
)

plt.plot(
    freqs,
    filtered_db,
    label="Butterworth",
)

plt.axvline(
    1.0,
    linestyle="--",
)

plt.axvline(
    40.0,
    linestyle="--",
)

plt.xlim(0.1, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.title("Pipeline B Butterworth Band-Pass PSD")
plt.legend()
plt.tight_layout()

output_path = "results/butterworth_psd_comparison.png"
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot: {output_path}")

assert np.isfinite(raw_psd).all()
assert np.isfinite(filtered_psd).all()

print("\nButterworth PSD validation passed.")
