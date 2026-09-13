import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROC_ROOT = ROOT / "data" / "processed"
DEFAULT_REPORTS_ROOT = ROOT / "reports" / "processed_summaries"


def build_summary_row(record_dir: Path, patient: str):
    X_path = record_dir.with_name(f"{record_dir.name}_X.npy")
    y_path = record_dir.with_name(f"{record_dir.name}_y.npy")
    meta_path = record_dir.with_name(f"{record_dir.name}_meta.json")

    if not all(p.exists() for p in (X_path, y_path, meta_path)):
        return None

    import numpy as np

    X = np.load(X_path)
    y = np.load(y_path)
    meta = json.loads(meta_path.read_text())

    record_id = record_dir.name
    seizure_count = len(meta.get("seizures", []))
    duration = meta.get("duration_seconds", len(y) * 2.0)

    return {
        "patient": patient,
        "record_id": record_id,
        "n_windows": int(len(y)),
        "n_ictal": int((y == 1).sum()),
        "n_interictal": int((y == 0).sum()),
        "n_unknown": int((y == -1).sum()),
        "duration_seconds": float(duration),
        "n_seizures": int(seizure_count),
        "channels": int(X.shape[1]),
        "feature_dim": int(X.shape[1]),
        "preprocessing_steps": "channel-selection -> FIR bandpass -> ICA artifact removal -> z-score normalization -> 256Hz resample -> 6s window stacking",
    }


def export_patient_summary(proc_dir: Path, reports_dir: Path):
    patient = proc_dir.name
    rows = []
    for f in sorted(proc_dir.glob("*_X.npy")):
        record_id = f.stem.replace("_X", "")
        row = build_summary_row(proc_dir / record_id, patient)
        if row is not None:
            rows.append(row)

    fieldnames = [
        "patient",
        "record_id",
        "n_windows",
        "n_ictal",
        "n_interictal",
        "n_unknown",
        "duration_seconds",
        "n_seizures",
        "channels",
        "feature_dim",
        "preprocessing_steps",
    ]

    out_csv = reports_dir / f"{patient}_preprocessed_summary.csv"
    with out_csv.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved patient summary: {out_csv} ({len(rows)} rows)")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Export processed EEG summaries for all processed patients.")
    ap.add_argument("--patient", default=None, help="Optional patient folder name, e.g. chb01")
    ap.add_argument("--proc-root", default=str(DEFAULT_PROC_ROOT))
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORTS_ROOT))
    args = ap.parse_args()

    proc_root = Path(args.proc_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.patient:
        patient_dirs = [proc_root / args.patient]
    else:
        patient_dirs = [p for p in sorted(proc_root.iterdir()) if p.is_dir()]

    all_rows = []
    for proc_dir in patient_dirs:
        if not proc_dir.exists():
            continue
        all_rows.extend(export_patient_summary(proc_dir, report_dir))

    combined_csv = report_dir / "all_patients_preprocessed_summary.csv"
    fieldnames = [
        "patient",
        "record_id",
        "n_windows",
        "n_ictal",
        "n_interictal",
        "n_unknown",
        "duration_seconds",
        "n_seizures",
        "channels",
        "feature_dim",
        "preprocessing_steps",
    ]

    with combined_csv.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved combined report: {combined_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
