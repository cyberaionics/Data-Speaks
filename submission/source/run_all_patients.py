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
REPORT_ROOT = ROOT / "reports"


def discover_patients(raw_root: Path):
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw patient root not found: {raw_root}")

    patients = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    if not patients:
        raise FileNotFoundError(f"No patient folders found in {raw_root}")
    return patients


def run_command(command):
    print(f"Running: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=str(ROOT), text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        return False
    return True


def selected_models(model, include_svm):
    models = [model] if model != "all" else ["logistic_regression", "random_forest", "knn"]
    if model == "all" and include_svm:
        models.append("svm")
    return models


def patient_has_benchmarks(patient, bench_root, models):
    return all((bench_root / f"{model}_{patient}_loso.json").exists() for model in models)


def main():
    ap = argparse.ArgumentParser(description="Batch preprocessing and model evaluation for all CHB-MIT patient folders.")
    ap.add_argument("--patients", nargs="*", default=None, help="Optional patient names to process, e.g. chb01 chb02")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--proc-root", default=str(PROC_ROOT))
    ap.add_argument("--bench-root", default=str(BENCH_ROOT))
    ap.add_argument("--prediction-root", default=str(PRED_ROOT))
    ap.add_argument("--fig-root", default=str(FIG_ROOT))
    ap.add_argument("--report-dir", default=str(REPORT_ROOT / "dataset_summary"))
    ap.add_argument("--processed-report-dir", default=str(REPORT_ROOT / "processed_summaries"))
    ap.add_argument("--model", choices=["logistic_regression", "random_forest", "knn", "svm", "all"], default="all")
    ap.add_argument("--include-svm", action="store_true", help="Include the slow RBF SVM when --model all is used")
    ap.add_argument("--resume", action="store_true", help="Skip patients whose requested benchmark files already exist")
    ap.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing and only run evaluation for already-processed patients")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    proc_root = Path(args.proc_root)
    bench_root = Path(args.bench_root)
    pred_root = Path(args.prediction_root)
    fig_root = Path(args.fig_root)
    report_dir = Path(args.report_dir)
    processed_report_dir = Path(args.processed_report_dir)

    proc_root.mkdir(parents=True, exist_ok=True)
    bench_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    patients = args.patients if args.patients else discover_patients(raw_root)
    print(f"Discovered patients: {patients}")
    failures = []
    models = selected_models(args.model, args.include_svm)

    for patient in patients:
        if args.resume and patient_has_benchmarks(patient, bench_root, models):
            print(f"Skipping completed patient {patient} (benchmarks already exist)")
            continue

        patient_raw_dir = raw_root / patient
        if not patient_raw_dir.exists():
            print(f"Skipping missing raw folder: {patient_raw_dir}")
            failures.append((patient, "missing raw folder"))
            continue

        if not args.skip_preprocess:
            if not run_command([
                sys.executable,
                str(SCRIPT_DIR / "preprocess_patient.py"),
                "--patient",
                patient,
                "--raw-root",
                str(raw_root),
                "--out-root",
                str(proc_root),
            ]):
                failures.append((patient, "preprocessing"))
                continue

        evaluated = True
        for model in models:
            if not run_command([
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
                model,
            ]):
                evaluated = False
                break
        if not evaluated:
            failures.append((patient, "model evaluation"))
            continue

        compared = run_command([
            sys.executable,
            str(SCRIPT_DIR / "compare_models.py"),
            "--patient",
            patient,
        ])
        if not compared:
            failures.append((patient, "model comparison"))

        reported = run_command([
            sys.executable,
            str(SCRIPT_DIR / "math_report.py"),
            "--patient",
            patient,
            "--proc-root",
            str(proc_root),
            "--fig-root",
            str(fig_root),
        ])
        if not reported:
            failures.append((patient, "math report"))

    run_command([
        sys.executable,
        str(SCRIPT_DIR / "export_midterm_dataset.py"),
        "--report-dir",
        str(processed_report_dir),
    ])

    run_command([
        sys.executable,
        str(SCRIPT_DIR / "summarize_all_patients.py"),
        "--bench-root",
        str(bench_root),
        "--fig-root",
        str(fig_root),
        "--report-dir",
        str(report_dir),
    ])

    if failures:
        print(f"Batch pipeline completed with {len(failures)} issue(s):")
        for patient, stage in failures:
            print(f"  {patient}: {stage}")
    else:
        print(f"Batch pipeline completed successfully for {len(patients)} patients.")


if __name__ == "__main__":
    main()
