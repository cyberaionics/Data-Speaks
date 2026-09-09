import gc
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline_d = load_module("pipeline_d", "pipeline/01_preprocessing/candidate_algorithms/pipeline_d.py")
stage0 = load_module("stage0", "pipeline/preprocessing_common/stage0.py")
RAW_DIR = Path("data/raw/physionet.org/chb02")
SUMMARY_PATH = RAW_DIR / "chb02-summary.txt"
OUTPUT_DIR = Path("data/processed/pipeline_D")
RESULTS_DIR = Path("results/benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
recordings = sorted(RAW_DIR.glob("*.edf"))
assert len(recordings) == 36, f"Expected 36 EDF recordings, found {len(recordings)}"


def json_converter(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


config = pipeline_d.PipelineDConfig()
results = []
for index, input_path in enumerate(recordings, start=1):
    recording_id = input_path.stem
    output_path = OUTPUT_DIR / f"{recording_id}_raw.fif"
    print(f"Processing {index}/{len(recordings)}: {recording_id}")
    gc.collect()
    raw, metadata = stage0.load_stage0(input_path, summary_file=SUMMARY_PATH)
    input_shape = raw.get_data().shape
    input_sfreq = float(raw.info["sfreq"])
    tracemalloc.start()
    start_time = time.perf_counter()
    processed, provenance = pipeline_d.run_pipeline_d(raw, config=config, recording_id=recording_id)
    runtime_sec = time.perf_counter() - start_time
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert processed.ch_names == raw.ch_names
    assert float(processed.info["sfreq"]) == config.target_sfreq
    assert processed.get_data().shape == input_shape
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
            "output_shape": processed.get_data().shape,
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

provenance_path = RESULTS_DIR / "pipeline_D_provenance.json"
with open(provenance_path, "w") as f:
    json.dump(results, f, indent=2, default=json_converter)
print("Pipeline D benchmark complete")
print("Recordings:", len(results))
print("Seizure recordings:", sum(bool(result["seizure_intervals"]) for result in results))
print("Total runtime:", sum(result["runtime_sec"] for result in results))
print("Provenance:", provenance_path)
