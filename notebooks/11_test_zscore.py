from pathlib import Path
import importlib.util

import numpy as np

from pipeline.preprocessing_common.stage0 import load_stage0


DATA_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_FILE = DATA_DIR / "chb02-summary.txt"


def load_function(file_path, function_name):
    spec = importlib.util.spec_from_file_location(
        function_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, function_name)


def main():
    zscore_function = load_function(
        Path("pipeline")
        / "01_preprocessing"
        / "candidate_algorithms"
        / "normalization"
        / "zscore.py",
        "apply_zscore",
    )

    raw, _ = load_stage0(
        DATA_DIR / "chb02_01.edf",
        SUMMARY_FILE,
    )

    # Use first 60 seconds for the test.
    raw.crop(tmin=0.0, tmax=60.0)

    normalized, provenance = zscore_function(raw)

    data = normalized.get_data()

    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)

    print("===== Z-SCORE RESULTS =====")
    print(f"Shape: {data.shape}")
    print(
        f"Maximum absolute channel mean: "
        f"{np.max(np.abs(means)):.6e}"
    )
    print(
        f"Minimum channel standard deviation: "
        f"{np.min(stds):.6f}"
    )
    print(
        f"Maximum channel standard deviation: "
        f"{np.max(stds):.6f}"
    )

    print("\nFirst five channels:")

    for i in range(min(5, len(normalized.ch_names))):
        print(
            f"{normalized.ch_names[i]:10s} "
            f"mean={means[i]: .6e} "
            f"std={stds[i]:.6f}"
        )


if __name__ == "__main__":
    main()
