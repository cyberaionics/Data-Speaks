from pathlib import Path

from pipeline.preprocessing_common.stage0 import load_stage0

import importlib.util
import sys

DATA_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_FILE = DATA_DIR / "chb02-summary.txt"


def load_pipeline_a():
    path = (
        Path("pipeline")
        / "01_preprocessing"
        / "candidate_algorithms"
        / "pipeline_a.py"
    )

    spec = importlib.util.spec_from_file_location(
        "pipeline_a_module",
        path,
    )
   
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_a_module"] = module
    spec.loader.exec_module(module)
    return (
        module.run_pipeline_a,
        module.PipelineAConfig,
    )


def main():
    run_pipeline_a, PipelineAConfig = load_pipeline_a()

    print("Loading Stage 0...")

    raw, metadata = load_stage0(
        DATA_DIR / "chb02_01.edf",
        SUMMARY_FILE,
    )

    # Short validation run.
    raw.crop(tmin=0.0, tmax=60.0)

    print("Stage 0 complete.")
    print(f"Input shape: {raw.get_data().shape}")
    print(f"Input sampling rate: {raw.info['sfreq']} Hz")

    config = PipelineAConfig()

    print("\nRunning complete Pipeline A...")

    processed, provenance = run_pipeline_a(
        raw,
        config,
        recording_id="chb02_01",
    )

    print("Pipeline A complete.")

    data = processed.get_data()

    print("\n===== FINAL OUTPUT =====")
    print(f"Shape: {data.shape}")
    print(f"Sampling rate: {processed.info['sfreq']} Hz")
    print(f"Channels: {len(processed.ch_names)}")
    print(f"Samples: {processed.n_times}")

    print("\n===== NORMALIZATION CHECK =====")

    import numpy as np

    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)

    print(
        f"Maximum absolute channel mean: "
        f"{np.max(np.abs(means)):.6e}"
    )

    print(
        f"Minimum channel std: "
        f"{np.min(stds):.6f}"
    )

    print(
        f"Maximum channel std: "
        f"{np.max(stds):.6f}"
    )

    print("\n===== PROVENANCE =====")

    print(
        f"Pipeline: "
        f"{provenance['pipeline']}"
    )

    print(
        f"FIR band: "
        f"{provenance['fir']['low_hz']}–"
        f"{provenance['fir']['high_hz']} Hz"
    )

    print(
        f"ICA components: "
        f"{provenance['ica']['actual_n_components']}"
    )

    print(
        f"ICA excluded: "
        f"{provenance['ica']['excluded_components']}"
    )

    print(
        f"Resampling operation: "
        f"{provenance['resampling']['operation']}"
    )

    # Save processed test result.
    output_dir = Path("results") / "pipeline_a_test"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "chb02_01_first60s_raw.fif"
    )

    processed.save(
        output_path,
        overwrite=True,
        verbose=False,
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
