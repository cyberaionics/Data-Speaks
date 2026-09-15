import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def discover_patients(bench_root: Path):
    patients = set()
    for path in bench_root.glob("*_loso.json"):
        if path.name.startswith("comparison_"):
            continue
        name = path.name
        for token in ["random_forest", "logistic_regression", "knn", "svm"]:
            if f"{token}_" in name:
                patient = name.replace(f"{token}_", "").replace("_loso.json", "")
                patients.add(patient)
                break
    for path in bench_root.glob("comparison_*.json"):
        patient = path.name.replace("comparison_", "").replace(".json", "")
        patients.add(patient)
    return sorted(patients)


def load_patient_ranking(patient: str, bench_root: Path):
    comparison_path = bench_root / f"comparison_{patient}.json"
    if comparison_path.exists():
        payload = json.loads(comparison_path.read_text())
        rows = payload.get("ranking", [])
        return rows

    rows = []
    for path in sorted(bench_root.glob(f"*_{patient}_loso.json")):
        if path.name.startswith("comparison_"):
            continue
        data = json.loads(path.read_text())
        summary = data.get("summary", {})
        if summary.get("n_records", 0) == 0:
            continue
        rows.append({"model": data.get("model"), **summary})

    rows.sort(key=lambda row: (
        -(row.get("sensitivity_mean") if row.get("sensitivity_mean") is not None else -1),
        row.get("fdr_per_24h_mean") if row.get("fdr_per_24h_mean") is not None else float("inf"),
        row.get("latency_sec_mean") if row.get("latency_sec_mean") is not None else float("inf"),
    ))
    return rows


def aggregate_model_metrics(patient_rankings):
    model_map = {}
    for patient_row in patient_rankings.values():
        for row in patient_row:
            model = row.get("model")
            if not model:
                continue
            entry = model_map.setdefault(model, {"patient_values": [], "sensitivity": [], "fdr": [], "latency": []})
            entry["patient_values"].append(row)
            if row.get("sensitivity_mean") is not None:
                entry["sensitivity"].append(float(row["sensitivity_mean"]))
            if row.get("fdr_per_24h_mean") is not None:
                entry["fdr"].append(float(row["fdr_per_24h_mean"]))
            if row.get("latency_sec_mean") is not None:
                entry["latency"].append(float(row["latency_sec_mean"]))

    overall = []
    for model, data in model_map.items():
        overall.append({
            "model": model,
            "patients_evaluated": len(data["patient_values"]),
            "mean_sensitivity": float(sum(data["sensitivity"]) / len(data["sensitivity"])) if data["sensitivity"] else None,
            "mean_fdr_per_24h": float(sum(data["fdr"]) / len(data["fdr"])) if data["fdr"] else None,
            "mean_latency_sec": float(sum(data["latency"]) / len(data["latency"])) if data["latency"] else None,
        })

    overall.sort(key=lambda row: (
        -(row.get("mean_sensitivity") if row.get("mean_sensitivity") is not None else -1),
        row.get("mean_fdr_per_24h") if row.get("mean_fdr_per_24h") is not None else float("inf"),
        row.get("mean_latency_sec") if row.get("mean_latency_sec") is not None else float("inf"),
    ))
    return overall


def save_model_summary_plot(model_summary, fig_root: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [row["model"] for row in model_summary]
    sensitivity = [row["mean_sensitivity"] for row in model_summary]
    fdr = [row["mean_fdr_per_24h"] for row in model_summary]
    latency = [row["mean_latency_sec"] for row in model_summary]

    axes[0].bar(labels, sensitivity, color="#4C78A8")
    axes[0].set_title("Mean sensitivity across patients")
    axes[0].set_ylabel("Sensitivity")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, fdr, color="#F58518")
    axes[1].set_title("Mean false detections / 24h")
    axes[1].set_ylabel("FDR / 24h")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, latency, color="#54A24B")
    axes[2].set_title("Mean latency (s)")
    axes[2].set_ylabel("Latency (s)")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Overall model comparison across processed patients")
    fig.tight_layout()
    out = fig_root / "overall_dataset_model_summary.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved dataset model summary plot: {out}")


def save_patient_summary_plot(patient_rankings, fig_root: Path):
    patients = list(patient_rankings)
    best_sensitivity = []
    for patient in patients:
        rows = patient_rankings[patient]
        if not rows:
            best_sensitivity.append((patient, None))
            continue
        best = rows[0]
        best_sensitivity.append((patient, best.get("sensitivity_mean")))

    labels = [p for p, _ in best_sensitivity]
    values = [v if v is not None else 0.0 for _, v in best_sensitivity]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#7A3E65")
    ax.set_title("Best sensitivity by patient")
    ax.set_ylabel("Sensitivity")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = fig_root / "overall_dataset_patient_best_sensitivity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved patient sensitivity plot: {out}")


def save_csv_summary(model_summary, patient_rankings, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "overall_dataset_model_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "patients_evaluated", "mean_sensitivity", "mean_fdr_per_24h", "mean_latency_sec"])
        writer.writeheader()
        writer.writerows(model_summary)

    with (report_dir / "overall_dataset_patient_ranking.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient", "model", "sensitivity_mean", "fdr_per_24h_mean", "latency_sec_mean"])
        writer.writeheader()
        for patient, rows in patient_rankings.items():
            for row in rows:
                writer.writerow({
                    "patient": patient,
                    "model": row.get("model"),
                    "sensitivity_mean": row.get("sensitivity_mean"),
                    "fdr_per_24h_mean": row.get("fdr_per_24h_mean"),
                    "latency_sec_mean": row.get("latency_sec_mean"),
                })


def main():
    ap = argparse.ArgumentParser(description="Summarize benchmark results across all processed patients and write overall figures/JSONs.")
    ap.add_argument("--bench-root", default=str(ROOT / "results" / "benchmarks"))
    ap.add_argument("--fig-root", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "dataset_summary"))
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    fig_root = Path(args.fig_root)
    report_dir = Path(args.report_dir)
    fig_root.mkdir(parents=True, exist_ok=True)

    patients = discover_patients(bench_root)
    if not patients:
        raise FileNotFoundError(f"No patient benchmark files found in {bench_root}")

    patient_rankings = {}
    for patient in patients:
        patient_rankings[patient] = load_patient_ranking(patient, bench_root)

    model_summary = aggregate_model_metrics(patient_rankings)

    overall_payload = {
        "n_patients": len(patients),
        "patients": patients,
        "overall_model_summary": model_summary,
        "per_patient_rankings": patient_rankings,
    }

    overall_json = bench_root / "overall_dataset_summary.json"
    overall_json.write_text(json.dumps(overall_payload, indent=2))
    print(f"Saved overall dataset summary: {overall_json}")

    save_model_summary_plot(model_summary, fig_root)
    save_patient_summary_plot(patient_rankings, fig_root)
    save_csv_summary(model_summary, patient_rankings, report_dir)

    print(f"Done: {len(patients)} patients aggregated into overall results.")


if __name__ == "__main__":
    main()
