import numpy as np
import pywt
import mne


def apply_wavelet_denoising(
    raw: mne.io.BaseRaw,
    wavelet: str = "db4",
    level: int = 5,
) -> tuple[mne.io.BaseRaw, dict]:
    """
    Apply wavelet soft-threshold denoising independently to
    each EEG channel.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Standardized EEG recording.
    wavelet : str
        Wavelet family/name used for decomposition.
    level : int
        Maximum decomposition level.

    Returns
    -------
    cleaned : mne.io.BaseRaw
        Wavelet-denoised EEG.
    provenance : dict
        Parameters and processing information.
    """
    if level < 1:
        raise ValueError("level must be at least 1")

    data = raw.get_data()

    if not np.isfinite(data).all():
        raise ValueError("Input EEG contains non-finite values")

    wavelet_obj = pywt.Wavelet(wavelet)

    max_level = pywt.dwt_max_level(
        data_len=data.shape[1],
        filter_len=wavelet_obj.dec_len,
    )

    if max_level < 1:
        raise ValueError(
            "Signal is too short for wavelet decomposition"
        )

    actual_level = min(level, max_level)

    denoised = np.empty_like(data)

    for channel_idx in range(data.shape[0]):
        signal = data[channel_idx]

        coeffs = pywt.wavedec(
            signal,
            wavelet=wavelet_obj,
            level=actual_level,
        )

        # Estimate noise sigma from the finest detail coefficients.
        finest_detail = coeffs[-1]

        mad = np.median(np.abs(finest_detail - np.median(finest_detail)))

        sigma = mad / 0.6745 if mad > 0 else 0.0

        # Universal threshold.
        threshold = (
            sigma * np.sqrt(2.0 * np.log(len(signal)))
            if sigma > 0
            else 0.0
        )

        thresholded_coeffs = [coeffs[0]]

        for detail in coeffs[1:]:
            thresholded_coeffs.append(
                pywt.threshold(
                    detail,
                    value=threshold,
                    mode="soft",
                )
            )

        reconstructed = pywt.waverec(
            thresholded_coeffs,
            wavelet=wavelet_obj,
        )

        # waverec can return one or two extra samples.
        denoised[channel_idx] = reconstructed[: signal.shape[0]]

    cleaned = mne.io.RawArray(
        denoised,
        raw.info.copy(),
        verbose=False,
    )

    provenance = {
        "method": "wavelet_denoising",
        "wavelet": wavelet,
        "requested_level": level,
        "actual_level": actual_level,
        "threshold": "universal_soft_threshold",
        "noise_estimator": "MAD",
    }

    return cleaned, provenance
