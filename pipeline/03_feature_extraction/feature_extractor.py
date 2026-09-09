import importlib.util
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


time_features = load_module(
    "time_domain_features",
    BASE_DIR / "time_domain" / "features.py",
)

frequency_features = load_module(
    "frequency_domain_features",
    BASE_DIR / "frequency_domain" / "features.py",
)

statistical_features = load_module(
    "statistical_features",
    BASE_DIR / "statistical" / "features.py",
)


def extract_features(
    window: np.ndarray,
    sfreq: float,
) -> dict:
    """
    Extract all common features from one EEG window.

    Returns a flat dictionary with one value per feature.
    """

    window = np.asarray(window, dtype=float)

    if window.ndim != 2:
        raise ValueError(
            "window must have shape (n_channels, n_samples)"
        )

    time = time_features.flatten_time_domain_features(window)

    frequency = (
        frequency_features.flatten_frequency_domain_features(
            window,
            sfreq,
        )
    )

    statistical = (
        statistical_features.flatten_statistical_features(
            window
        )
    )

    combined = {}
    combined.update(time)
    combined.update(frequency)
    combined.update(statistical)

    return combined
