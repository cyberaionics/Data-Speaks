import numpy as np

from features import (
    extract_time_domain_features,
    flatten_time_domain_features,
)


def main():
    # Small synthetic EEG window:
    # 2 channels, 8 samples
    window = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ],
        dtype=float,
    )

    features = extract_time_domain_features(window)

    assert features["mean"].shape == (2,)
    assert features["std"].shape == (2,)
    assert features["min"].shape == (2,)
    assert features["max"].shape == (2,)
    assert features["ptp"].shape == (2,)
    assert features["rms"].shape == (2,)
    assert features["mean_abs"].shape == (2,)
    assert features["zero_crossing_rate"].shape == (2,)

    # First channel: 1..8
    assert np.isclose(features["mean"][0], 4.5)
    assert np.isclose(features["min"][0], 1.0)
    assert np.isclose(features["max"][0], 8.0)
    assert np.isclose(features["ptp"][0], 7.0)

    # Second channel changes sign at every step
    assert np.isclose(features["zero_crossing_rate"][1], 1.0)

    flattened = flatten_time_domain_features(window)

    assert len(flattened) == 16
    assert "ch1_mean" in flattened
    assert "ch2_rms" in flattened
    assert "ch2_zero_crossing_rate" in flattened

    print("All time-domain feature tests passed.")


if __name__ == "__main__":
    main()
