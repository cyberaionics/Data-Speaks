import argparse
from pathlib import Path
import sys, json
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score

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
    candidates = []
    for f in sorted(proc_dir.glob("*_X.npy")):
        rec_id = f.stem.replace("_X", "")
        X = np.load(f)
        candidates.append({
            "id": rec_id,
            "X": X,
            "y": np.load(proc_dir / f"{rec_id}_y.npy"),
            "starts": np.load(proc_dir / f"{rec_id}_starts.npy"),
            "meta": json.loads((proc_dir / f"{rec_id}_meta.json").read_text()),
            "width": int(X.shape[1]),
        })

    if not candidates:
        return {}

    # Keep only the most common feature width so LOSO stacking is consistent.
    widths = {}
    for c in candidates:
        widths[c["width"]] = widths.get(c["width"], 0) + 1
    majority_width = max(widths, key=widths.get)
    kept = [c for c in candidates if c["width"] == majority_width]
    dropped = [c["id"] for c in candidates if c["width"] != majority_width]
    if dropped:
        print(f"  Dropping {len(dropped)} record(s) with non-majority feature width {widths}: {dropped}")

    return {c["id"]: {"X": c["X"], "y": c["y"], "starts": c["starts"], "meta": c["meta"]} for c in kept}


def evaluate_model(records, train, predict, prediction_dir):
    results = []
    for test_id, y_pred, y_true, y_score in leave_one_record_out(records, train, predict):
        rec = records[test_id]
        np.save(prediction_dir / f"{test_id}_y_pred.npy", y_pred)
        np.save(prediction_dir / f"{test_id}_y_score.npy", y_score)
        events = detect_events(y_pred, rec["starts"])
        seizures = rec["meta"]["seizures"]
        duration = rec["meta"].get("duration_seconds", rec["meta"]["n_windows"] * 2.0)
        row = {
            "test_record": test_id,
            "n_windows": int(len(y_true)),
            "n_positive_windows": int((y_pred == 1).sum()),
            "n_events": len(events),
            "epoch_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "_y_true": y_true,
            "_y_pred": y_pred,
            "_y_score": y_score,
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
    y_true = np.concatenate([row["_y_true"] for row in results]) if results else np.array([])
    y_pred = np.concatenate([row["_y_pred"] for row in results]) if results else np.array([])
    y_score = np.concatenate([row["_y_score"] for row in results]) if results else np.array([])
    return {
        "n_records": len(results),
        "n_seizure_records": len(seizure_rows),
        "n_nonseizure_records": len(nonseizure_rows),
        "sensitivity_mean": float(sum(row["sensitivity"] for row in seizure_rows) / len(seizure_rows)) if seizure_rows else None,
        "epoch_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
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
    print(f"Patient {args.patient}: {len(records)} records", flush=True)

    selected = list(MODELS) if args.model == "all" else [args.model]
    for model_name in selected:
        print(f"Starting model {model_name} for {args.patient}", flush=True)
        train, predict = load_model(model_name)
        prediction_dir = prediction_root / model_name / args.patient
        prediction_dir.mkdir(parents=True, exist_ok=True)
        results = evaluate_model(records, train, predict, prediction_dir)
        serializable_records = [{key: value for key, value in row.items() if not key.startswith("_")} for row in results]
        output = {"model": model_name, "patient": args.patient, "summary": summarize(results), "records": serializable_records}
        out = bench_dir / f"{model_name}_{args.patient}_loso.json"
        out.write_text(json.dumps(output, indent=2))
        print(f"{model_name}: {output['summary']}", flush=True)
        print(f"Saved -> {out}", flush=True)


if __name__ == "__main__":
    main()