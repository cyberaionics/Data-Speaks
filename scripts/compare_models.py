import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Rank classifier benchmark results for one patient.")
    ap.add_argument("--patient", default="chb01")
    ap.add_argument("--bench-root", default=str(ROOT / "results" / "benchmarks"))
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    rows = []
    for path in sorted(bench_root.glob(f"*_{args.patient}_loso.json")):
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            continue
        summary = result.get("summary", {})
        if summary.get("n_records", 0) == 0:
            continue
        rows.append({"model": result["model"], **summary})

    rows.sort(key=lambda row: (
        -(row["sensitivity_mean"] if row["sensitivity_mean"] is not None else -1),
        row["fdr_per_24h_mean"] if row["fdr_per_24h_mean"] is not None else float("inf"),
        row["latency_sec_mean"] if row["latency_sec_mean"] is not None else float("inf"),
    ))
    output = bench_root / f"comparison_{args.patient}.json"
    output.write_text(json.dumps({"patient": args.patient, "ranking": rows}, indent=2))
    print(f"Patient {args.patient} model ranking:")
    for index, row in enumerate(rows, 1):
        print(f"{index}. {row['model']}: sensitivity={row['sensitivity_mean']}, "
              f"FDR/24h={row['fdr_per_24h_mean']}, latency={row['latency_sec_mean']}s")
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()