import numpy as np


def extract_statistical_features(window: np.ndarray) -> dict:
    """
    Extract statistical features from one EEG window.

    Parameters
    ----------
    window : np.ndarray
        EEG data with shape (n_channels, n_samples).

    Returns
    -------
    dict
        Per-channel statistical features.
    """
    window = np.asarray(window, dtype=float)

    if window.ndim != 2:
        raise ValueError("window must have shape (n_channels, n_samples)")

    if window.shape[1] < 2:
        raise ValueError("window must contain at least 2 samples")

    features = {}

    # Median
    features["median"] = np.median(window, axis=1)

    # Mean absolute deviation from the channel mean
    mean = np.mean(window, axis=1, keepdims=True)
    features["mean_abs_deviation"] = np.mean(
        np.abs(window - mean),
        axis=1,
    )

    # Median absolute deviation
    median = np.median(window, axis=1, keepdims=True)
    features["median_abs_deviation"] = np.median(
        np.abs(window - median),
        axis=1,
    )

    # Interquartile range
    q25 = np.percentile(window, 25, axis=1)
    q75 = np.percentile(window, 75, axis=1)
    features["iqr"] = q75 - q25

    # Robust amplitude range
    features["q05_q95_range"] = (
        np.percentile(window, 95, axis=1)
        - np.percentile(window, 5, axis=1)
    )

    # Skewness calculated directly from central moments
    centered = window - mean
    std = np.std(window, axis=1)

    third_moment = np.mean(centered ** 3, axis=1)

    features["skewness"] = np.divide(
        third_moment,
        std ** 3,
        out=np.zeros_like(third_moment),
        where=std > 0,
    )

    # Excess kurtosis
    fourth_moment = np.mean(centered ** 4, axis=1)

    features["excess_kurtosis"] = np.divide(
        fourth_moment,
        std ** 4,
        out=np.zeros_like(fourth_moment),
        where=std > 0,
    ) - 3.0

    return features


def flatten_statistical_features(window: np.ndarray) -> dict:
    """
    Flatten statistical features into channel-feature names.
    """
    features = extract_statistical_features(window)

    flattened = {}

    for feature_name, values in features.items():
        for channel_idx, value in enumerate(values):
            flattened[
                f"ch{channel_idx + 1}_{feature_name}"
            ] = float(value)

    return flattened
