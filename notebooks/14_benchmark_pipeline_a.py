import gc
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.preprocessing_common.stage0 import load_stage0


RAW_DIR = ROOT / "data/raw/physionet.org/chb02"
SUMMARY_PATH = RAW_DIR / "chb02-summary.txt"
OUTPUT_DIR = ROOT / "data/processed/pipeline_A"
RESULTS_DIR = ROOT / "results/benchmarks"
PROVENANCE_PATH = RESULTS_DIR / "pipeline_A_provenance.json"


def load_pipeline_a():
    path = ROOT / "pipeline/01_preprocessing/candidate_algorithms/pipeline_a.py"
    spec = importlib.util.spec_from_file_location("pipeline_a_full", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def json_converter(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    pipeline_a = load_pipeline_a()
    recordings = sorted(RAW_DIR.glob("*.edf"))
    if not recordings:
        raise RuntimeError(f"No EDF recordings found in {RAW_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    config = pipeline_a.PipelineAConfig()

    for index, input_path in enumerate(recordings, start=1):
        recording_id = input_path.stem
        output_path = OUTPUT_DIR / f"{recording_id}_raw.fif"
        print(f"Processing {index}/{len(recordings)}: {recording_id}")
        gc.collect()

        raw, metadata = load_stage0(input_path, summary_file=SUMMARY_PATH)
        input_shape = raw.get_data().shape
        input_sfreq = float(raw.info["sfreq"])
        tracemalloc.start()
        start_time = time.perf_counter()
        processed, provenance = pipeline_a.run_pipeline_a(
            raw, config=config, recording_id=recording_id
        )
        runtime_sec = time.perf_counter() - start_time
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        output_data = processed.get_data()
        if processed.ch_names != raw.ch_names:
            raise RuntimeError(f"Channel contract failed for {recording_id}")
        if float(processed.info["sfreq"]) != config.target_sfreq:
            raise RuntimeError(f"Sampling-rate contract failed for {recording_id}")
        if output_data.shape != input_shape:
            raise RuntimeError(f"Shape contract failed for {recording_id}: {output_data.shape}")
        if not np.isfinite(output_data).all():
            raise RuntimeError(f"Non-finite output for {recording_id}")

        processed.set_meas_date(None)
        processed.save(output_path, overwrite=True, verbose=False)
        results.append(
            {
                "recording_id": recording_id,
                "input_file": str(input_path),
                "output_file": str(output_path),
                "duration_sec": raw.n_times / input_sfreq,
                "input_sfreq": input_sfreq,
                "output_sfreq": float(processed.info["sfreq"]),
                "input_shape": input_shape,
                "output_shape": output_data.shape,
                "n_channels": len(processed.ch_names),
                "channel_names": list(processed.ch_names),
                "seizure_intervals": metadata.seizure_intervals,
                "runtime_sec": runtime_sec,
                "peak_tracemalloc_mb": peak_memory_bytes / (1024 ** 2),
                "provenance": provenance,
            }
        )
        print(f"  runtime: {runtime_sec:.2f} sec")
        del processed, raw
        gc.collect()

    with PROVENANCE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=json_converter)

    print("Pipeline A full benchmark complete")
    print("Recordings:", len(results))
    print("Seizure recordings:", sum(bool(item["seizure_intervals"]) for item in results))
    print("Total runtime:", sum(item["runtime_sec"] for item in results))
    print("Provenance:", PROVENANCE_PATH)


if __name__ == "__main__":
    main()
