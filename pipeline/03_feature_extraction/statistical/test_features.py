import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "statistical_features_test_module",
    Path(__file__).with_name("features.py"),
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load statistical features module")
features_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features_module)

extract_statistical_features = features_module.extract_statistical_features
flatten_statistical_features = features_module.flatten_statistical_features


def main():
    window = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ],
        dtype=float,
    )

    features = extract_statistical_features(window)

    expected_features = [
        "median",
        "mean_abs_deviation",
        "median_abs_deviation",
        "iqr",
        "q05_q95_range",
        "skewness",
        "excess_kurtosis",
    ]

    for name in expected_features:
        assert name in features
        assert features[name].shape == (2,)
        assert np.all(np.isfinite(features[name]))

    # First channel: 1..8
    assert np.isclose(features["median"][0], 4.5)
    assert np.isclose(features["iqr"][0], 3.5)

    # Symmetric sequences should have approximately zero skewness.
    assert np.isclose(features["skewness"][0], 0.0)

    flattened = flatten_statistical_features(window)

    assert len(flattened) == 2 * len(expected_features)
    assert "ch1_median" in flattened
    assert "ch2_iqr" in flattened
    assert "ch1_skewness" in flattened
    assert "ch2_excess_kurtosis" in flattened

    print("All statistical feature tests passed.")


if __name__ == "__main__":
    main()
