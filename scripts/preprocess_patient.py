import argparse
from pathlib import Path
import sys, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: F401

from stage0 import load_and_select_18
from pipeline_a import run_pipeline
from windowing import segment_epochs, stack_epochs, EPOCH_SEC, STACK_W
from extract_features import extract
from labels import parse_seizure_file, label_windows


def process_record(edf_path, out_dir):
    rec_id = edf_path.stem
    print(f"  {rec_id} ...", end=" ", flush=True)

    raw_std, s0 = load_and_select_18(edf_path)
    raw_out, prov = run_pipeline(raw_std, recording_id=rec_id)
    data = raw_out.get_data().astype(np.float32)
    sfreq = float(raw_out.info["sfreq"])

    epochs, e_starts = segment_epochs(data, sfreq)
    stacked, s_starts = stack_epochs(epochs, e_starts)
    if len(stacked) == 0:
        print("no windows")
        return None

    X = extract(stacked, sfreq, sets=("spectral",))
    seizures = parse_seizure_file(edf_path)
    y = label_windows(s_starts, EPOCH_SEC, STACK_W, seizures)

    keep = y != -1
    X, y, s_starts = X[keep], y[keep], s_starts[keep]

    np.save(out_dir / f"{rec_id}_X.npy", X)
    np.save(out_dir / f"{rec_id}_y.npy", y)
    np.save(out_dir / f"{rec_id}_starts.npy", s_starts)
    (out_dir / f"{rec_id}_meta.json").write_text(json.dumps({"recording_id": rec_id, "sfreq": sfreq, "n_windows": int(len(X)), "n_ictal": int((y == 1).sum()), "n_interictal": int((y == 0).sum()), "duration_seconds": float(data.shape[1] / sfreq), "seizures": seizures, "stage0": s0, "pipeline": prov}, indent=2, default=str))

    print(f"{len(X)} win (ictal={int((y==1).sum())})")
    return rec_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="chb01")
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw" / "physionet.org"))
    ap.add_argument("--out-root", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    patient_dir = Path(args.raw_root) / args.patient
    out_dir = Path(args.out_root) / args.patient
    out_dir.mkdir(parents=True, exist_ok=True)

    edfs = sorted(patient_dir.glob("*.edf"))
    print(f"Patient {args.patient}: {len(edfs)} EDFs → {out_dir}")
    for edf in edfs:
        try:
            process_record(edf, out_dir)
        except Exception as e:
            print(f"  {edf.stem}: FAILED — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()