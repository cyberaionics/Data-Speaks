import time
from dataclasses import dataclass, field, asdict

from mne.io import BaseRaw

from filtering.fir_bandpass import fir_bandpass, DEFAULT_L_FREQ, DEFAULT_H_FREQ
from artifact_removal.ica import fit_ica, select_artifact_components, apply_ica, DEFAULT_N_COMPONENTS, DEFAULT_RANDOM_STATE
from normalization.zscore import zscore_normalize
from resampling.resample_256 import resample_to_target, TARGET_SFREQ

PIPELINE_ID = "pipeline_A_fir_ica_zscore_256"


@dataclass
class PipelineAConfig:
    l_freq: float = DEFAULT_L_FREQ
    h_freq: float = DEFAULT_H_FREQ
    n_components: int = DEFAULT_N_COMPONENTS
    random_state: int = DEFAULT_RANDOM_STATE
    eog_proxy: str = "Fp1"
    eog_corr_thresh: float = 0.4
    muscle_thresh: float = 0.3
    target_sfreq: float = TARGET_SFREQ
    n_jobs: int = 1
    extra: dict = field(default_factory=dict)


def run_pipeline_a(raw_std: BaseRaw, config: PipelineAConfig | None = None, recording_id: str | None = None):
    t0 = time.perf_counter()
    cfg = config or PipelineAConfig()
    provenance: dict = {"pipeline_id": PIPELINE_ID,"recording_id": recording_id, "parameters": asdict(cfg), "input_channels": list(raw_std.ch_names), "input_sfreq": raw_std.info["sfreq"], "input_n_times": raw_std.n_times}

    raw_filt = fir_bandpass(raw_std, cfg.l_freq, cfg.h_freq, cfg.n_jobs)

    ica = fit_ica(raw_filt, cfg.n_components, random_state=cfg.random_state)
    comp_scores = select_artifact_components(ica, raw_filt, eog_proxy=cfg.eog_proxy, eog_corr_thresh=cfg.eog_corr_thresh, muscle_thresh=cfg.muscle_thresh)
    raw_clean = apply_ica(raw_filt, ica, comp_scores["excluded_components"])
    provenance["ica"] = comp_scores
    provenance["ica_n_components_fitted"] = ica.n_components_

    raw_z, norm_stats = zscore_normalize(raw_clean)
    provenance["normalization"] = norm_stats

    raw_out, was_resampled = resample_to_target(raw_z, cfg.target_sfreq)
    provenance["resampled"] = was_resampled
    provenance["output_sfreq"] = raw_out.info["sfreq"]

    provenance["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    return raw_out, provenance


if __name__ == "__main__":
    import json
    import re
    import traceback
    from pathlib import Path

    import mne

    DATA_ROOT = Path("/run/media/rangerofdanger/Files/Data Speaks/data/raw/physionet.org/")
    OUT_DIR = Path("/run/media/rangerofdanger/Files/Data Speaks/data/processed/pipeline_A")
    PROV_DIR = Path("/run/media/rangerofdanger/Files/Data Speaks/results/benchmarks/pipeline_A")
    PATIENT_DIVISOR = 4
    OVERRIDE_PATIENT = "chb01"

    def selected_patient_dirs(root: Path) -> list[Path]:
        dirs = []
        for d in sorted(root.iterdir()):
            m = re.fullmatch(r"chb(\d{2})", d.name)
            if d.is_dir() and m and int(m.group(1)) % PATIENT_DIVISOR == 0:
                dirs.append(d)
        return dirs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROV_DIR.mkdir(parents=True, exist_ok=True)

    run_summary = {"pipeline_id": PIPELINE_ID, "patients": [], "failures": []}

    patient_dirs = ([DATA_ROOT / OVERRIDE_PATIENT] if OVERRIDE_PATIENT else selected_patient_dirs(DATA_ROOT))
    missing = [str(d) for d in patient_dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"Patient folder(s) not found under {DATA_ROOT}: {missing}. ""Check OVERRIDE_PATIENT / DATA_ROOT.")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        run_summary["patients"].append(patient_id)
        print(f"=== {patient_id} ===")

        for edf_path in sorted(patient_dir.glob("*.edf")):
            recording_id = edf_path.stem
            try:
                raw_std = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raw_out, prov = run_pipeline_a(raw_std,recording_id=recording_id)
                prov["stage0_bypassed"] = True 
                raw_out.save(OUT_DIR / f"{recording_id}_raw.fif",overwrite=True, verbose=False)
                with open(PROV_DIR / f"{recording_id}_provenance.json","w") as fh:
                    json.dump(prov, fh, indent=2)

                print(f"  OK   {recording_id} " f"({prov['runtime_seconds']}s, " f"ICs excluded: {prov['ica']['excluded_components']})")
            except Exception:
                run_summary["failures"].append({"recording_id": recording_id,"traceback": traceback.format_exc()})
                print(f"  FAIL {recording_id} — see run_summary.json")

        with open(PROV_DIR / "run_summary.json", "w") as fh:
            json.dump(run_summary, fh, indent=2)