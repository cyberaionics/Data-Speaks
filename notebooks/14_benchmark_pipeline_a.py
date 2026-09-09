from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import resource
import time

from pipeline.preprocessing_common.stage0 import load_stage0


DATA_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_FILE = DATA_DIR / "chb02-summary.txt"
OUTPUT_DIR = Path("data/processed/pipeline_A")
PROVENANCE_PATH = (
    Path("results/benchmarks")
    / "pipeline_A_provenance.json"
)

def json_converter(obj):
    if hasattr(obj, "item"):
        return obj.item()

    if hasattr(obj, "tolist"):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )

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

    import sys

    sys.modules["pipeline_a_module"] = module
    spec.loader.exec_module(module)

    return module.run_pipeline_a, module.PipelineAConfig


def get_peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)

    # macOS reports ru_maxrss in bytes.
    return usage.ru_maxrss / (1024 * 1024)


def main():
    run_pipeline_a, PipelineAConfig = load_pipeline_a()

    recordings = [
        "chb02_01.edf",
        "chb02_16.edf",
    ]

    config = PipelineAConfig()

    results = []

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path("results/benchmarks").mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in recordings:
        print("\n" + "=" * 70)
        print(f"BENCHMARKING: {filename}")
        print("=" * 70)

        edf_path = DATA_DIR / filename

        print("Loading Stage 0...")

        raw, metadata = load_stage0(
            edf_path,
            SUMMARY_FILE,
        )

        print("Stage 0 complete.")

        print(
            f"Input: {metadata.n_channels} channels, "
            f"{metadata.sampling_rate_hz} Hz, "
            f"{metadata.duration_sec:.2f} sec"
        )

        start_time = time.perf_counter()

        processed, provenance = run_pipeline_a(
            raw,
            config,
            recording_id=metadata.recording_id,
        )

        elapsed = time.perf_counter() - start_time

        output_path = (
            OUTPUT_DIR
            / f"{metadata.recording_id}_raw.fif"
        )

        print("Saving processed FIF...")

        processed.save(
            output_path,
            overwrite=True,
            verbose=False,
        )

        result = {
            "recording_id": metadata.recording_id,
            "source_file": str(edf_path),
            "duration_sec": metadata.duration_sec,
            "input_channels": metadata.n_channels,
            "input_sampling_rate_hz": metadata.sampling_rate_hz,
            "output_channels": len(processed.ch_names),
            "output_samples": processed.n_times,
            "output_sampling_rate_hz": float(
                processed.info["sfreq"]
            ),
            "runtime_sec": elapsed,
            "peak_memory_mb": get_peak_memory_mb(),
            "seizure_intervals": metadata.seizure_intervals,
            "ica_excluded_components": provenance[
                "ica"
            ]["excluded_components"],
            "output_path": str(output_path),
            "pipeline": provenance["pipeline"],
            "provenance": provenance,
        }

        results.append(result)

        print("\nResult:")
        print(f"Runtime: {elapsed:.2f} sec")
        print(
            f"Peak memory: "
            f"{result['peak_memory_mb']:.2f} MB"
        )
        print(
            f"Output shape: "
            f"{processed.get_data().shape}"
        )
        print(
            f"ICA excluded: "
            f"{result['ica_excluded_components']}"
        )
        print(f"Saved: {output_path}")

        # Release large objects before next recording.
        del raw
        del processed

    with PROVENANCE_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
	    default=json_converter,
        )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Provenance: {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
