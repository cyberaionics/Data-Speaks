import argparse
from pathlib import Path
import sys, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline

from svm_model import train as svm_train, predict as svm_predict
from loso import leave_one_record_out
from metrics import detect_events, sensitivity, latency, false_detection_per_24h


def load_patient(proc_dir):
    records = {}
    for f in sorted(proc_dir.glob("*_X.npy")):
        rec_id = f.stem.replace("_X", "")
        records[rec_id] = {"X": np.load(f), "y": np.load(proc_dir / f"{rec_id}_y.npy"), "starts": np.load(proc_dir / f"{rec_id}_starts.npy"), "meta": json.loads((proc_dir / f"{rec_id}_meta.json").read_text())}
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="chb01")
    ap.add_argument("--proc-root", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--bench-root", default=str(ROOT / "results" / "benchmarks"))
    args = ap.parse_args()

    proc_dir = Path(args.proc_root) / args.patient
    bench_dir = Path(args.bench_root); bench_dir.mkdir(parents=True, exist_ok=True)

    records = load_patient(proc_dir)
    print(f"Patient {args.patient}: {len(records)} records")

    results = []
    for test_id, y_pred, y_true in leave_one_record_out(
            records, svm_train, svm_predict):
        rec = records[test_id]
        events = detect_events(y_pred, rec["starts"])
        sz = rec["meta"]["seizures"]
        dur = rec["meta"].get("duration_seconds", rec["meta"]["n_windows"] * 2.0)
        row = {"test_record": test_id, "n_events": len(events)}
        if sz:
            row["sensitivity"] = sensitivity(sz, events)
            row["latency_sec"] = latency(sz, events)
        else:
            row["fdr_per_24h"] = false_detection_per_24h(events, dur)
        results.append(row)
        print(f"  {test_id}: {row}")

    out = bench_dir / f"pipeline_a_{args.patient}_loso.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()