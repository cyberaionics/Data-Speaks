"""
DATA-SPEAKS — Pipeline C chb01 Batch Processing Runner
======================================================
Processes all recordings for patient chb01 through Stage 0 and Pipeline C.
Generates an aggregated CSV of evaluation metrics saved to:
results/tables/pipeline_c_chb01_metrics.csv
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PREPROC_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(PREPROC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from stage0 import run_stage0, DATASET_DIR
from pipeline_c import run_pipeline_c
from pipeline_c_metrics import build_csv_row


def main():
    print("=" * 70)
    print("STARTING PIPELINE C RUN FOR PATIENT chb01")
    print("=" * 70)

    chb01_dir = DATASET_DIR / "chb01"
    summary_path = chb01_dir / "chb01-summary.txt"
    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "pipeline_c_chb01_metrics.csv"

    # Find all EDF files in chb01
    edf_files = sorted(list(chb01_dir.glob("*.edf")))
    print(f"Found {len(edf_files)} EDF files in {chb01_dir}")

    csv_rows = []
    failed_recordings = []

    for idx, edf_path in enumerate(edf_files, start=1):
        rec_id = edf_path.stem
        print(f"\n[{idx}/{len(edf_files)}] Processing {rec_id}...")
        t_start = time.time()

        try:
            # 1. Stage 0
            s0_res = run_stage0(
                edf_path=edf_path,
                summary_path=summary_path,
                patient_id="chb01",
                recording_id=rec_id,
            )

            # Check if candidate ready (skip CS2 special montages or incomplete standard montages if policy requires)
            if not s0_res.candidate_ready:
                print(f"  [Warning] {rec_id} candidate_ready is False (special_montage={s0_res.is_special_montage}). Proceeding with available core channels.")

            # 2. Pipeline C
            c_out = run_pipeline_c(
                raw=s0_res.raw,
                seizure_intervals=s0_res.seizure_intervals,
                asr_cutoff=20.0,
            )

            # 3. Extract metrics row
            row = build_csv_row(
                patient_id="chb01",
                recording_id=rec_id,
                stage0_result=s0_res,
                pipeline_c_outputs=c_out,
            )
            csv_rows.append(row)
            t_elapsed = time.time() - t_start
            print(f"  [Success] {rec_id} processed in {t_elapsed:.1f}s")

            # Incrementally write/update CSV so progress is saved
            if csv_rows:
                fieldnames = list(csv_rows[0].keys())
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_rows)

        except Exception as e:
            t_elapsed = time.time() - t_start
            err_msg = traceback.format_exc()
            print(f"  [FAILED] {rec_id} failed after {t_elapsed:.1f}s: {e}")
            failed_recordings.append({
                "recording_id": rec_id,
                "error": str(e),
                "traceback": err_msg,
            })

    print("\n" + "=" * 70)
    print("PIPELINE C chb01 BATCH COMPLETE")
    print(f"Total processed: {len(csv_rows)} / {len(edf_files)}")
    print(f"Failures: {len(failed_recordings)}")
    print(f"Metrics table saved to: {out_csv}")
    print("=" * 70)

    if failed_recordings:
        print("\nSummary of failures:")
        for fail in failed_recordings:
            print(f"  - {fail['recording_id']}: {fail['error']}")


if __name__ == "__main__":
    main()
