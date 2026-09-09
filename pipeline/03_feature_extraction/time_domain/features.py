import numpy as np


def extract_time_domain_features(window: np.ndarray) -> dict:
    """
    Extract time-domain features from one EEG window.

    Parameters
    ----------
    window : np.ndarray
        EEG data with shape (n_channels, n_samples).

    Returns
    -------
    dict
        Per-channel time-domain features.
    """
    window = np.asarray(window, dtype=float)

    if window.ndim != 2:
        raise ValueError("window must have shape (n_channels, n_samples)")

    if window.shape[1] < 2:
        raise ValueError("window must contain at least 2 samples")

    features = {}

    # Basic amplitude statistics
    features["mean"] = np.mean(window, axis=1)
    features["std"] = np.std(window, axis=1)
    features["min"] = np.min(window, axis=1)
    features["max"] = np.max(window, axis=1)

    # Amplitude range
    features["ptp"] = np.ptp(window, axis=1)

    # Root Mean Square
    features["rms"] = np.sqrt(np.mean(window ** 2, axis=1))

    # Mean absolute amplitude
    features["mean_abs"] = np.mean(np.abs(window), axis=1)

    # Zero-crossing rate
    signs = np.signbit(window)
    zero_crossings = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    features["zero_crossing_rate"] = (
        zero_crossings / (window.shape[1] - 1)
    )

    return features


def flatten_time_domain_features(window: np.ndarray) -> dict:
    """
    Extract time-domain features and flatten them into
    channel-feature names suitable for a feature table.
    """
    features = extract_time_domain_features(window)

    flattened = {}

    for feature_name, values in features.items():
        for channel_idx, value in enumerate(values):
            flattened[f"ch{channel_idx + 1}_{feature_name}"] = float(value)

    return flattened
