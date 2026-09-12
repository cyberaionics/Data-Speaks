import argparse
from pathlib import Path
import sys, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline

from loso import leave_one_record_out
from metrics import detect_events, sensitivity, latency, false_detection_per_24h

MODELS = {
    "svm": ("svm_model", "pipeline_a"),
    "logistic_regression": ("lr_model", "logistic_regression"),
    "random_forest": ("rf_model", "random_forest"),
    "knn": ("knn_model", "knn"),
}


def load_patient(proc_dir):
    records = {}
    for f in sorted(proc_dir.glob("*_X.npy")):
        rec_id = f.stem.replace("_X", "")
        records[rec_id] = {"X": np.load(f), "y": np.load(proc_dir / f"{rec_id}_y.npy"), "starts": np.load(proc_dir / f"{rec_id}_starts.npy"), "meta": json.loads((proc_dir / f"{rec_id}_meta.json").read_text())}
    return records


def evaluate_model(records, train, predict, prediction_dir):
    results = []
    for test_id, y_pred, y_true in leave_one_record_out(records, train, predict):
        rec = records[test_id]
        np.save(prediction_dir / f"{test_id}_y_pred.npy", y_pred)
        events = detect_events(y_pred, rec["starts"])
        seizures = rec["meta"]["seizures"]
        duration = rec["meta"].get("duration_seconds", rec["meta"]["n_windows"] * 2.0)
        row = {
            "test_record": test_id,
            "n_windows": int(len(y_true)),
            "n_positive_windows": int((y_pred == 1).sum()),
            "n_events": len(events),
        }
        if seizures:
            row["sensitivity"] = sensitivity(seizures, events)
            row["latency_sec"] = latency(seizures, events)
        else:
            row["fdr_per_24h"] = false_detection_per_24h(events, duration)
        results.append(row)
    return results


def summarize(results):
    seizure_rows = [row for row in results if "sensitivity" in row]
    nonseizure_rows = [row for row in results if "fdr_per_24h" in row]
    latencies = [row["latency_sec"] for row in seizure_rows if row["latency_sec"] is not None]
    return {
        "n_records": len(results),
        "n_seizure_records": len(seizure_rows),
        "n_nonseizure_records": len(nonseizure_rows),
        "sensitivity_mean": float(sum(row["sensitivity"] for row in seizure_rows) / len(seizure_rows)) if seizure_rows else None,
        "latency_sec_mean": float(sum(latencies) / len(latencies)) if latencies else None,
        "fdr_per_24h_mean": float(sum(row["fdr_per_24h"] for row in nonseizure_rows) / len(nonseizure_rows)) if nonseizure_rows else None,
    }


def load_model(model_name):
    module_name, _ = MODELS[model_name]
    module = __import__(module_name)
    return module.train, module.predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="chb01")
    ap.add_argument("--proc-root", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--bench-root", default=str(ROOT / "results" / "benchmarks"))
    ap.add_argument("--prediction-root", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--model", choices=[*MODELS, "all"], default="logistic_regression")
    args = ap.parse_args()

    proc_dir = Path(args.proc_root) / args.patient
    bench_dir = Path(args.bench_root); bench_dir.mkdir(parents=True, exist_ok=True)
    prediction_root = Path(args.prediction_root)

    records = load_patient(proc_dir)
    print(f"Patient {args.patient}: {len(records)} records")

    selected = list(MODELS) if args.model == "all" else [args.model]
    for model_name in selected:
        train, predict = load_model(model_name)
        prediction_dir = prediction_root / model_name / args.patient
        prediction_dir.mkdir(parents=True, exist_ok=True)
        results = evaluate_model(records, train, predict, prediction_dir)
        output = {"model": model_name, "patient": args.patient, "summary": summarize(results), "records": results}
        out = bench_dir / f"{model_name}_{args.patient}_loso.json"
        out.write_text(json.dumps(output, indent=2))
        print(f"{model_name}: {output['summary']}")
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()