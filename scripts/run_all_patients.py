import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
RAW_ROOT = ROOT / "data" / "raw" / "physionet.org"
PROC_ROOT = ROOT / "data" / "processed"
BENCH_ROOT = ROOT / "results" / "benchmarks"
PRED_ROOT = ROOT / "results" / "predictions"
FIG_ROOT = ROOT / "results" / "figures"


def discover_patients(raw_root: Path):
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw patient root not found: {raw_root}")

    patients = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    if not patients:
        raise FileNotFoundError(f"No patient folders found in {raw_root}")
    return patients


def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=str(ROOT), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def main():
    ap = argparse.ArgumentParser(description="Batch preprocessing and model evaluation for all CHB-MIT patient folders.")
    ap.add_argument("--patients", nargs="*", default=None, help="Optional patient names to process, e.g. chb01 chb02")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--proc-root", default=str(PROC_ROOT))
    ap.add_argument("--bench-root", default=str(BENCH_ROOT))
    ap.add_argument("--prediction-root", default=str(PRED_ROOT))
    ap.add_argument("--fig-root", default=str(FIG_ROOT))
    ap.add_argument("--model", choices=["logistic_regression", "random_forest", "knn", "svm", "all"], default="all")
    ap.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing and only run evaluation for already-processed patients")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    proc_root = Path(args.proc_root)
    bench_root = Path(args.bench_root)
    pred_root = Path(args.prediction_root)
    fig_root = Path(args.fig_root)

    proc_root.mkdir(parents=True, exist_ok=True)
    bench_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    patients = args.patients if args.patients else discover_patients(raw_root)
    print(f"Discovered patients: {patients}")

    for patient in patients:
        patient_raw_dir = raw_root / patient
        if not patient_raw_dir.exists():
            print(f"Skipping missing raw folder: {patient_raw_dir}")
            continue

        if not args.skip_preprocess:
            run_command([
                sys.executable,
                str(SCRIPT_DIR / "preprocess_patient.py"),
                "--patient",
                patient,
                "--raw-root",
                str(raw_root),
                "--out-root",
                str(proc_root),
            ])

        if args.model == "all":
            run_command([
                sys.executable,
                str(SCRIPT_DIR / "run_patient.py"),
                "--patient",
                patient,
                "--proc-root",
                str(proc_root),
                "--bench-root",
                str(bench_root),
                "--prediction-root",
                str(pred_root),
                "--model",
                "all",
            ])
        else:
            run_command([
                sys.executable,
                str(SCRIPT_DIR / "run_patient.py"),
                "--patient",
                patient,
                "--proc-root",
                str(proc_root),
                "--bench-root",
                str(bench_root),
                "--prediction-root",
                str(pred_root),
                "--model",
                args.model,
            ])

        run_command([
            sys.executable,
            str(SCRIPT_DIR / "compare_models.py"),
            "--patient",
            patient,
        ])

        run_command([
            sys.executable,
            str(SCRIPT_DIR / "math_report.py"),
            "--patient",
            patient,
            "--proc-root",
            str(proc_root),
            "--fig-root",
            str(fig_root),
        ])

    print("Batch pipeline setup complete. Run this script when you are ready to process all patients.")


if __name__ == "__main__":
    main()
