import numpy as np

from feature_extractor import extract_features


def main():
    sfreq = 256.0

    # 23 channels × 1024 samples = one real project window shape
    rng = np.random.default_rng(42)

    window = rng.normal(
        0,
        1,
        size=(23, 1024),
    )

    features = extract_features(
        window,
        sfreq,
    )

    expected_count = 23 * (8 + 11 + 7)

    print("Feature count:", len(features))
    print("Expected:", expected_count)

    assert len(features) == expected_count
    assert all(
        np.isfinite(value)
        for value in features.values()
    )

    assert "ch1_mean" in features
    assert "ch1_alpha_power" in features
    assert "ch1_median" in features

    print("Common feature extractor test passed.")


if __name__ == "__main__":
    main()
