import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"ictal": "#d1495b", "interictal": "#30638e"}


def load_json(path):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else None


def save_class_balance(proc_dir, fig_dir, patient):
    ictal = 0
    interictal = 0
    records = []
    for label_path in sorted(proc_dir.glob("*_y.npy")):
        y = np.load(label_path)
        record = label_path.stem.removesuffix("_y")
        n_ictal = int((y == 1).sum())
        n_interictal = int((y == 0).sum())
        ictal += n_ictal
        interictal += n_interictal
        records.append((record, n_ictal, n_interictal))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["Interictal", "Ictal"], [interictal, ictal], color=[COLORS["interictal"], COLORS["ictal"]])
    axes[0].set_ylabel("Windows")
    axes[0].set_title(f"Patient {patient}: class balance")
    axes[0].set_yscale("log")
    axes[0].text(0, interictal, f"{interictal:,}", ha="center", va="bottom")
    axes[0].text(1, ictal, f"{ictal:,}", ha="center", va="bottom")

    names = [row[0].replace(f"{patient}_", "") for row in records]
    axes[1].bar(names, [row[2] for row in records], color=COLORS["interictal"], label="Interictal")
    axes[1].bar(names, [row[1] for row in records], bottom=[row[2] for row in records], color=COLORS["ictal"], label="Ictal")
    axes[1].set_title("Windows by recording")
    axes[1].set_xlabel("Recording")
    axes[1].set_ylabel("Windows")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{patient}_class_balance.png", dpi=200)
    plt.close(fig)


def save_model_comparison(bench_dir, fig_dir, patient):
    comparison = load_json(bench_dir / f"comparison_{patient}.json")
    if not comparison or not comparison.get("ranking"):
        return
    rows = comparison["ranking"]
    names = [row["model"].replace("_", " ").title() for row in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [
        ("sensitivity_mean", "Sensitivity", "higher", (0, 1.05)),
        ("fdr_per_24h_mean", "False detections / 24h", "lower", None),
        ("latency_sec_mean", "Mean latency (s)", "lower", None),
    ]
    for ax, (key, title, _, ylim) in zip(axes, metrics):
        values = [row.get(key) if row.get(key) is not None else np.nan for row in rows]
        bars = ax.bar(x, values, color=["#00798c", "#edae49", "#d1495b"][:len(values)])
        ax.set_title(title)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Patient {patient}: classifier comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{patient}_model_comparison.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(bench_dir, prediction_root, proc_dir, fig_dir, patient, model):
    y_true_parts = []
    y_pred_parts = []
    for label_path in sorted(proc_dir.glob("*_y.npy")):
        record = label_path.stem.removesuffix("_y")
        prediction_path = prediction_root / model / patient / f"{record}_y_pred.npy"
        if not prediction_path.exists():
            continue
        y_true = np.load(label_path)
        y_pred = np.load(prediction_path)
        if len(y_true) == len(y_pred):
            y_true_parts.append(y_true)
            y_pred_parts.append(y_pred)
    if not y_true_parts:
        return
    matrix = confusion_matrix(np.hstack(y_true_parts), np.hstack(y_pred_parts), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Interictal", "Ictal"])
    ax.set_yticks([0, 1], ["Interictal", "Ictal"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model.replace('_', ' ').title()} confusion matrix")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, label="Windows")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{patient}_{model}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_prediction_timeline(bench_dir, prediction_root, proc_dir, fig_dir, patient, model, record):
    prediction_path = prediction_root / model / patient / f"{record}_y_pred.npy"
    label_path = proc_dir / f"{record}_y.npy"
    starts_path = proc_dir / f"{record}_starts.npy"
    meta_path = proc_dir / f"{record}_meta.json"
    if not all(path.exists() for path in (prediction_path, label_path, starts_path, meta_path)):
        return
    y_pred = np.load(prediction_path)
    y_true = np.load(label_path)
    starts = np.load(starts_path)
    meta = json.loads(meta_path.read_text())
    n = min(len(y_pred), len(y_true), len(starts))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(starts[:n], y_true[:n], where="post", color="#30638e", alpha=0.75, label="True label")
    ax.step(starts[:n], y_pred[:n] + 0.04, where="post", color="#d1495b", alpha=0.75, label="Prediction")
    for start, end in meta.get("seizures", []):
        ax.axvspan(start, end, color="#edae49", alpha=0.25, label="Annotated seizure")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([0, 1], ["Interictal", "Ictal"])
    ax.set_ylim(-0.2, 1.35)
    ax.set_title(f"{patient} {record}: {model.replace('_', ' ').title()} prediction timeline")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{patient}_{model}_{record}_timeline.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Create interpretable plots from processed data and predictions.")
    ap.add_argument("--patient", default="chb01")
    ap.add_argument("--model", default="logistic_regression")
    ap.add_argument("--record", default=None, help="Record for timeline plot; defaults to first available prediction.")
    ap.add_argument("--proc-root", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--bench-root", default=str(ROOT / "results" / "benchmarks"))
    ap.add_argument("--prediction-root", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--fig-root", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()

    proc_dir = Path(args.proc_root) / args.patient
    bench_dir = Path(args.bench_root)
    prediction_root = Path(args.prediction_root)
    fig_dir = Path(args.fig_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_class_balance(proc_dir, fig_dir, args.patient)
    save_model_comparison(bench_dir, fig_dir, args.patient)
    save_confusion_matrix(bench_dir, prediction_root, proc_dir, fig_dir, args.patient, args.model)
    record = args.record
    if record is None:
        available = sorted((prediction_root / args.model / args.patient).glob("*_y_pred.npy"))
        record = available[0].name.removesuffix("_y_pred.npy") if available else None
    if record:
        save_prediction_timeline(bench_dir, prediction_root, proc_dir, fig_dir, args.patient, args.model, record)
    print(f"Saved interpretation figures -> {fig_dir}")


if __name__ == "__main__":
    main()