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


# =========================================================
# Load modules
# =========================================================

pipeline_b = load_module(
    "pipeline_b",
    "pipeline/01_preprocessing/candidate_algorithms/pipeline_b.py",
)

stage0 = load_module(
    "stage0",
    "pipeline/preprocessing_common/stage0.py",
)


# =========================================================
# Paths
# =========================================================

RAW_DIR = Path(
    "data/raw/physionet.org/chb02"
)

SUMMARY_PATH = RAW_DIR / "chb02-summary.txt"

OUTPUT_DIR = Path(
    "data/processed/pipeline_B"
)

RESULTS_DIR = Path(
    "results/benchmarks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Automatically discover EDF recordings
# =========================================================

RECORDINGS = sorted(
    RAW_DIR.glob("*.edf")
)

print(
    f"Found {len(RECORDINGS)} EDF recordings."
)

assert len(RECORDINGS) == 36, (
    f"Expected 36 EDF recordings, "
    f"found {len(RECORDINGS)}"
)


# =========================================================
# JSON converter
# =========================================================

def json_converter(value):
    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "tolist"):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        f"is not JSON serializable"
    )


# =========================================================
# Pipeline configuration
# =========================================================

config = pipeline_b.PipelineBConfig()


# =========================================================
# Results
# =========================================================

results = []


# =========================================================
# Run benchmark
# =========================================================

for index, input_path in enumerate(RECORDINGS, start=1):

    recording_id = input_path.stem

    output_path = (
        OUTPUT_DIR
        / f"{recording_id}_raw.fif"
    )

    print("\n" + "=" * 70)
    print(
        f"Processing {index}/{len(RECORDINGS)}: "
        f"{recording_id}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Memory cleanup
    # -----------------------------------------------------

    gc.collect()

    # -----------------------------------------------------
    # Stage 0
    # -----------------------------------------------------

    raw, metadata = stage0.load_stage0(
        input_path,
        summary_file=SUMMARY_PATH,
    )

    input_shape = raw.get_data().shape

    input_sfreq = float(
        raw.info["sfreq"]
    )

    duration_sec = (
        raw.n_times / input_sfreq
    )

    seizure_intervals = (
        metadata.seizure_intervals
    )

    print("Input shape:", input_shape)
    print(
        "Input sampling rate:",
        input_sfreq,
    )
    print(
        "Duration:",
        duration_sec,
    )
    print(
        "Channels:",
        len(raw.ch_names),
    )
    print(
        "Seizure intervals:",
        seizure_intervals,
    )

    # -----------------------------------------------------
    # Pipeline B
    # -----------------------------------------------------

    gc.collect()

    tracemalloc.start()

    start_time = time.perf_counter()

    processed, provenance = (
        pipeline_b.run_pipeline_b(
            raw,
            config=config,
            recording_id=recording_id,
        )
    )

    runtime_sec = (
        time.perf_counter()
        - start_time
    )

    current_memory, peak_memory_bytes = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    peak_memory_mb = (
        peak_memory_bytes
        / (1024 ** 2)
    )

    # -----------------------------------------------------
    # Output information
    # -----------------------------------------------------

    output_shape = (
        processed.get_data().shape
    )

    output_sfreq = float(
        processed.info["sfreq"]
    )

    output_channels = list(
        processed.ch_names
    )

    # -----------------------------------------------------
    # FIF compatibility
    # -----------------------------------------------------

    processed.set_meas_date(None)

    processed.save(
        output_path,
        overwrite=True,
        verbose=False,
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    assert (
        len(processed.ch_names)
        == len(raw.ch_names)
    )

    assert (
        processed.ch_names
        == raw.ch_names
    )

    assert (
        output_sfreq
        == config.target_sfreq
    )

    assert (
        output_shape
        == input_shape
    )

    assert (
        processed.get_data().shape
        == input_shape
    )

    # -----------------------------------------------------
    # Store result
    # -----------------------------------------------------

    result = {
        "recording_id": recording_id,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "duration_sec": duration_sec,
        "input_sfreq": input_sfreq,
        "output_sfreq": output_sfreq,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "n_channels": len(output_channels),
        "channel_names": output_channels,
        "seizure_intervals": seizure_intervals,
        "runtime_sec": runtime_sec,
        "peak_tracemalloc_mb": peak_memory_mb,
        "provenance": provenance,
    }

    results.append(result)

    # -----------------------------------------------------
    # Print recording result
    # -----------------------------------------------------

    print(
        "Output shape:",
        output_shape,
    )

    print(
        "Output sampling rate:",
        output_sfreq,
    )

    print(
        f"Runtime: {runtime_sec:.2f} sec"
    )

    print(
        f"Peak tracemalloc memory: "
        f"{peak_memory_mb:.2f} MB"
    )

    print(
        "Saved:",
        output_path,
    )

    # -----------------------------------------------------
    # Release memory
    # -----------------------------------------------------

    del processed
    del raw

    gc.collect()


# =========================================================
# Save complete provenance
# =========================================================

provenance_path = (
    RESULTS_DIR
    / "pipeline_B_provenance.json"
)

with open(
    provenance_path,
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=2,
        default=json_converter,
    )


# =========================================================
# Final summary
# =========================================================

print("\n" + "=" * 70)
print("Pipeline B COMPLETE CHB02 benchmark")
print("=" * 70)

print(
    "Recordings processed:",
    len(results),
)

print(
    "Recordings with seizures:",
    sum(
        bool(result["seizure_intervals"])
        for result in results
    ),
)

total_runtime = sum(
    result["runtime_sec"]
    for result in results
)

print(
    f"Total preprocessing runtime: "
    f"{total_runtime:.2f} sec"
)

print("\nPer-recording results:")

for result in results:

    print(
        f"{result['recording_id']}: "
        f"{result['runtime_sec']:.2f} sec, "
        f"{result['peak_tracemalloc_mb']:.2f} MB, "
        f"seizures={result['seizure_intervals']}"
    )

print(
    "\nProvenance:",
    provenance_path,
)

print("\nAll Pipeline B recordings processed successfully.")
