import numpy as np

from features import (
    extract_frequency_domain_features,
    flatten_frequency_domain_features,
)


def main():
    sfreq = 256.0

    # 2 channels, 4 seconds = 1024 samples
    t = np.arange(1024) / sfreq

    # Channel 1: 10 Hz sine wave
    # Channel 2: 20 Hz sine wave
    window = np.array(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 20 * t),
        ]
    )

    features = extract_frequency_domain_features(window, sfreq)

    expected_features = [
        "delta_power",
        "delta_relative_power",
        "theta_power",
        "theta_relative_power",
        "alpha_power",
        "alpha_relative_power",
        "beta_power",
        "beta_relative_power",
        "gamma_power",
        "gamma_relative_power",
        "spectral_centroid",
    ]

    for name in expected_features:
        assert name in features
        assert features[name].shape == (2,)
        assert np.all(np.isfinite(features[name]))

    # 10 Hz should dominate the alpha band for channel 1.
    assert features["alpha_relative_power"][0] > 0.5

    # 20 Hz should dominate the beta band for channel 2.
    assert features["beta_relative_power"][1] > 0.5

    flattened = flatten_frequency_domain_features(window, sfreq)

    assert len(flattened) == 2 * len(expected_features)
    assert "ch1_alpha_power" in flattened
    assert "ch2_beta_relative_power" in flattened
    assert "ch1_spectral_centroid" in flattened

    print("All frequency-domain feature tests passed.")


if __name__ == "__main__":
    main()
