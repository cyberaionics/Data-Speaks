import argparse
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: F401

from pca_reduce import pca_project
from math_metrics import cluster_separation, band_power_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="chb01")
    ap.add_argument("--proc-root", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--fig-root", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()

    proc_dir = Path(args.proc_root) / args.patient
    fig_dir = Path(args.fig_root); fig_dir.mkdir(parents=True, exist_ok=True)

    Xs, ys = [], []
    for f in sorted(proc_dir.glob("*_X.npy")):
        rec_id = f.stem.replace("_X", "")
        Xs.append(np.load(f))
        ys.append(np.load(proc_dir / f"{rec_id}_y.npy"))
    if not Xs:
        print(f"No processed records found for {args.patient} in {proc_dir}")
        return
    X = np.vstack(Xs); y = np.hstack(ys)
    print(f"{len(X)} windows ({int((y==1).sum())} ictal, " f"{int((y==0).sum())} interictal)")

    Z, _, evr = pca_project(X, n_components=2)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.4, label="Interictal", color="steelblue")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=6, alpha=0.6, label="Ictal", color="crimson")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.set_title(f"PCA of spectral features — {args.patient}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.patient}_pca.png", dpi=200)
    plt.close(fig)
    sil = cluster_separation(Z, y)
    bands, m1, m0, delta = band_power_shift(X, y)
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.4
    ax.bar(bands - w / 2, m0, w, label="Interictal", color="steelblue")
    ax.bar(bands + w / 2, m1, w, label="Ictal", color="crimson")
    ax.set_xlabel("Spectral band index (0.5→25 Hz)")
    ax.set_ylabel("Mean band energy")
    ax.set_title(f"Per-band energy shift — {args.patient}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.patient}_band_shift.png", dpi=200)
    plt.close(fig)

    summary = {"patient": args.patient, "n_windows": int(len(X)), "n_ictal": int((y == 1).sum()), "n_interictal": int((y == 0).sum()), "pca_explained_variance": evr.tolist(), "silhouette_pca2d": sil, "band_shift": delta.tolist()}
    (fig_dir / f"{args.patient}_math_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved → {fig_dir}")


if __name__ == "__main__":
    main()