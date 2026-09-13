import numpy as np


EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def extract_frequency_domain_features(
    window: np.ndarray,
    sfreq: float,
) -> dict:
    """
    Extract frequency-domain features from one EEG window.

    Parameters
    ----------
    window : np.ndarray
        EEG data with shape (n_channels, n_samples).
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Per-channel spectral features.
    """
    window = np.asarray(window, dtype=float)

    if window.ndim != 2:
        raise ValueError("window must have shape (n_channels, n_samples)")

    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    n_channels, n_samples = window.shape

    # Remove DC component before calculating the spectrum.
    centered = window - np.mean(window, axis=1, keepdims=True)

    # Real FFT.
    spectrum = np.fft.rfft(centered, axis=1)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)

    # Power spectral density-like quantity.
    power = (np.abs(spectrum) ** 2) / n_samples

    features = {}

    total_mask = (frequencies >= 1.0) & (frequencies <= 40.0)
    total_power = np.sum(power[:, total_mask], axis=1)

    for band_name, (low_freq, high_freq) in EEG_BANDS.items():
        mask = (frequencies >= low_freq) & (frequencies < high_freq)

        if not np.any(mask):
            raise ValueError(
                f"No frequency bins found for {band_name} band"
            )

        band_power = np.sum(power[:, mask], axis=1)

        features[f"{band_name}_power"] = band_power

        # Avoid division by zero for pathological windows.
        relative_power = np.divide(
            band_power,
            total_power,
            out=np.zeros_like(band_power),
            where=total_power > 0,
        )

        features[f"{band_name}_relative_power"] = relative_power

    # Spectral centroid over the 1–40 Hz range.
    frequencies_2d = frequencies[total_mask][None, :]
    power_total = power[:, total_mask]

    spectral_centroid = np.divide(
        np.sum(power_total * frequencies_2d, axis=1),
        total_power,
        out=np.zeros(n_channels),
        where=total_power > 0,
    )

    features["spectral_centroid"] = spectral_centroid

    return features


def flatten_frequency_domain_features(
    window: np.ndarray,
    sfreq: float,
) -> dict:
    """
    Flatten per-channel frequency-domain features into
    channel-feature names.
    """
    features = extract_frequency_domain_features(window, sfreq)

    flattened = {}

    for feature_name, values in features.items():
        for channel_idx, value in enumerate(values):
            flattened[
                f"ch{channel_idx + 1}_{feature_name}"
            ] = float(value)

    return flattened
