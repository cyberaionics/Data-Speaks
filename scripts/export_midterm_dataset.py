import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_ROOT = ROOT / "data" / "processed" / "chb01"
OUT_CSV = PROC_ROOT / "chb01_preprocessed_summary.csv"


def build_summary_row(record_dir: Path):
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
        "patient": "chb01",
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


def main():
    rows = []
    for f in sorted(PROC_ROOT.glob("*_X.npy")):
        record_id = f.stem.replace("_X", "")
        row = build_summary_row(Path(PROC_ROOT) / record_id)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved processed dataset summary CSV: {OUT_CSV}")
    print(f"Rows: {len(df)}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
