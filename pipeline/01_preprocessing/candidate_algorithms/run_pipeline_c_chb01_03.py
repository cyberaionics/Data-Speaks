"""
DATA-SPEAKS — Pipeline C Single Recording Validation Runner
============================================================
Runs Stage 0 and Pipeline C on chb01_03.edf.
Verifies the ASRpy patch, generates metrics, and produces validation plots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

# Add parent directories to sys.path to allow imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PREPROC_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(PREPROC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from stage0 import run_stage0, DATASET_DIR
from pipeline_c import run_pipeline_c
from pipeline_c_metrics import (
    compute_structural_metrics,
    compute_signal_stats,
    compute_spectral_metrics,
    compute_qc_metrics,
    compute_ica_metrics,
    compute_data_preservation_metrics,
    build_csv_row,
)
from pipeline_c_plots import (
    plot_time_domain_comparison,
    plot_spectral_and_diagnostics,
)


def main():
    print("=" * 70)
    print("RUNNING PIPELINE C VALIDATION ON chb01_03")
    print("=" * 70)

    edf_path = DATASET_DIR / "chb01" / "chb01_03.edf"
    summary_path = DATASET_DIR / "chb01" / "chb01-summary.txt"

    if not edf_path.exists():
        print(f"Error: EDF file not found at {edf_path}")
        sys.exit(1)

    # 1. Run Stage 0
    print("\n--- Step 1: Stage 0 Preprocessing ---")
    s0_res = run_stage0(
        edf_path=edf_path,
        summary_path=summary_path,
        patient_id="chb01",
        recording_id="chb01_03",
    )
    print(f"Stage 0 complete: {len(s0_res.channels_kept)} channels, duration {s0_res.duration_sec:.1f}s")
    print(f"Seizure intervals: {[f'{iv.start_sec}-{iv.end_sec}s' for iv in s0_res.seizure_intervals]}")

    # 2. Run Pipeline C
    print("\n--- Step 2: Pipeline C Preprocessing ---")
    c_out = run_pipeline_c(
        raw=s0_res.raw,
        seizure_intervals=s0_res.seizure_intervals,
        asr_cutoff=20.0,
    )
    timings = c_out["timings"]
    print("Pipeline C completed successfully!")
    print(f"Timings: Total = {timings['total_pipeline_c_sec']:.2f}s "
          f"(Butterworth: {timings['butterworth_sec']:.2f}s, "
          f"ASR: {timings['asr_sec']:.2f}s, "
          f"ICA: {timings['ica_sec']:.2f}s, "
          f"Std: {timings['standardization_sec']:.2f}s)")

    # 3. Compute Metrics
    print("\n--- Step 3: Computing Metrics ---")
    struct = compute_structural_metrics(s0_res.raw, c_out["final"])
    ica_m = compute_ica_metrics(c_out["ica"], s0_res.raw)
    pres_m = compute_data_preservation_metrics(
        s0_res.raw,
        raw_filtered=c_out["filtered"],
        raw_asr=c_out["asr_cleaned"],
        raw_ica=c_out["ica_cleaned"],
        raw_final=c_out["final"],
    )
    spec_m = compute_spectral_metrics(s0_res.raw, c_out["final"], all_channels=True)
    qc_m = compute_qc_metrics(s0_res.channel_qc)

    print(f"Output NaNs: {struct['output_nan_count']}, Infs: {struct['output_inf_count']}")
    print(f"ICA components fit: {ica_m['n_ica_components_fit']}, excluded: {ica_m['n_components_excluded']}")
    print(f"Relative RMS change (final): {pres_m['final']['relative_rms_change']:.4f}")

    # 4. Generate Validation Plots
    print("\n--- Step 4: Generating Validation Plots ---")
    fig1 = plot_time_domain_comparison(
        raw_stage0=s0_res.raw,
        raw_filtered=c_out["filtered"],
        raw_asr=c_out["asr_cleaned"],
        raw_final=c_out["final"],
        t_start=2990.0,
        t_end=3045.0,
    )
    fig2 = plot_spectral_and_diagnostics(
        raw_stage0=s0_res.raw,
        raw_final=c_out["final"],
        ica=c_out["ica"],
        timings=timings,
    )

    # 5. Save Single Record JSON benchmark summary
    bench_dir = REPO_ROOT / "results" / "benchmarks" / "pipeline_C"
    bench_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "pipeline_id": "pipeline_C_butterworth_asr_ica_zscore_256",
        "patient_id": "chb01",
        "recording_id": "chb01_03",
        "timings": timings,
        "structural": struct,
        "ica": ica_m,
        "data_preservation": pres_m,
        "qc_summary": qc_m.get("totals", {}),
    }
    def _numpy_json_default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(bench_dir / "chb01_03_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=_numpy_json_default)
    print(f"[Results] Saved summary JSON to {bench_dir / 'chb01_03_summary.json'}")

    print("\n" + "=" * 70)
    print("chb01_03 VALIDATION RUN FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
