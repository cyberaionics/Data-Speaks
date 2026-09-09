from pathlib import Path
import importlib.util

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
    resample_function = load_function(
        Path("pipeline")
        / "01_preprocessing"
        / "candidate_algorithms"
        / "resampling"
        / "resample_256.py",
        "resample_to_256",
    )

    raw, _ = load_stage0(
        DATA_DIR / "chb02_01.edf",
        SUMMARY_FILE,
    )

    # Short test segment.
    raw.crop(tmin=0.0, tmax=60.0)

    print("Before:")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Samples: {raw.n_times}")

    output, provenance = resample_function(raw)

    print("\nAfter:")
    print(f"Sampling rate: {output.info['sfreq']} Hz")
    print(f"Samples: {output.n_times}")

    print("\nProvenance:")
    for key, value in provenance.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
