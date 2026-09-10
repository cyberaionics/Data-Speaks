"""Pipeline A orchestrator — Conservative / Classical reference pipeline.

    FIR band-pass (1-40 Hz)
        -> ICA artifact removal
        -> per-channel Z-score
        -> 256 Hz

Contract (identical for all candidate pipelines):
    Input  : standardized EEG (mne.io.Raw) produced by Common Stage 0
             + one row of metadata.csv identifying the recording.
    Output : (processed Raw, provenance dict).

    The orchestrator NEVER modifies its input Raw; each substep returns a
    copy. All parameters and run statistics are recorded in the provenance
    dict, which the shared evaluator and metadata.csv consume.

Files are named after the algorithm they implement, per the current
team naming convention.
"""

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
from mne.io import BaseRaw

# FIF stores timestamps as signed 32-bit seconds. CHB-MIT anonymizes EDF
# startdates as year 2076, which overflows that range and makes raw.save() fail.
_FIF_SECS_MIN = np.iinfo(">i4").min  # -2147483648
_FIF_SECS_MAX = np.iinfo(">i4").max  #  2147483647

from filtering.fir_bandpass import fir_bandpass, DEFAULT_L_FREQ, DEFAULT_H_FREQ
from artifact_removal.ica import (
    fit_ica, select_artifact_components, apply_ica,
    DEFAULT_N_COMPONENTS, DEFAULT_RANDOM_STATE,
)
from normalization.zscore import zscore_normalize
from resampling.resample_256 import resample_to_target, TARGET_SFREQ

PIPELINE_ID = "pipeline_A_fir_ica_zscore_256"


@dataclass
class PipelineAConfig:
    """Every parameter of Pipeline A, in one auditable place.

    Changing a default here changes what the shared evaluator measures;
    document the reason for any change in the git commit message.
    """
    l_freq: float = DEFAULT_L_FREQ
    h_freq: float = DEFAULT_H_FREQ
    n_components: int = DEFAULT_N_COMPONENTS
    random_state: int = DEFAULT_RANDOM_STATE
    eog_proxy: str = "auto"   # resolved against channel names in ica.py
    eog_corr_thresh: float = 0.4
    muscle_thresh: float = 0.3
    target_sfreq: float = TARGET_SFREQ
    n_jobs: int = 1
    extra: dict = field(default_factory=dict)


def _datetime_to_secs(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, (tuple, list)) and value:
        return int(value[0])
    if isinstance(value, dict) and "secs" in value:
        return int(value["secs"])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _secs_out_of_fif_range(secs: int | None) -> bool:
    return secs is not None and (secs < _FIF_SECS_MIN or secs > _FIF_SECS_MAX)


def to_jsonable(obj):
    """Convert numpy scalars/arrays so provenance dumps with stdlib json."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sanitize_fif_timestamps(raw: BaseRaw) -> dict:
    """Make a Raw object writable as FIF.

    CHB-MIT EDF headers use two-digit years that MNE interprets as 2076.
    Those Unix timestamps exceed FIF's int32 meas_date field. The original
    dates are already fake (anonymized), so clearing them does not lose
    clinically meaningful information. Relative sample timing is unchanged.

    Returns a small audit dict for provenance.
    """
    info = raw.info
    original = info.get("meas_date")
    original_secs = _datetime_to_secs(original)
    cleared = False

    if _secs_out_of_fif_range(original_secs):
        raw.set_meas_date(None)
        cleared = True

    for key in ("file_id", "meas_id"):
        rec = info.get(key)
        if isinstance(rec, dict) and _secs_out_of_fif_range(_datetime_to_secs(rec)):
            rec["secs"] = 0
            if "usecs" in rec:
                rec["usecs"] = 0
            cleared = True

    for record in info.get("proc_history") or []:
        if isinstance(record, dict) and _secs_out_of_fif_range(
            _datetime_to_secs(record.get("date"))
        ):
            record["date"] = (0, 0)
            cleared = True
        block_id = record.get("block_id") if isinstance(record, dict) else None
        if isinstance(block_id, dict) and _secs_out_of_fif_range(
            _datetime_to_secs(block_id)
        ):
            block_id["secs"] = 0
            if "usecs" in block_id:
                block_id["usecs"] = 0
            cleared = True

    return {
        "cleared": cleared,
        "original_meas_date": str(original) if original is not None else None,
        "original_meas_date_secs": original_secs,
        "reason": (
            "CHB-MIT anonymized EDF date exceeds FIF int32 range"
            if cleared
            else None
        ),
    }


def run_pipeline_a(raw_std: BaseRaw, config: PipelineAConfig | None = None,
                   recording_id: str | None = None):
    """Run the full Pipeline A on one standardized recording.

    Returns
    -------
    raw_out : mne.io.BaseRaw
        Processed EEG at 256 Hz, Z-scored, artifact-reduced.
    provenance : dict
        Everything the shared evaluator / metadata.csv needs: runtime,
        parameters, ICA exclusion audit trail, normalization statistics,
        QC summary passthrough.
    """
    t0 = time.perf_counter()
    cfg = config or PipelineAConfig()
    provenance: dict = {
        "pipeline_id": PIPELINE_ID,
        "recording_id": recording_id,
        "parameters": asdict(cfg),
        "input_channels": list(raw_std.ch_names),
        "input_sfreq": raw_std.info["sfreq"],
        "input_n_times": raw_std.n_times,
    }

    # 1. FIR band-pass
    raw_filt = fir_bandpass(raw_std, cfg.l_freq, cfg.h_freq, cfg.n_jobs)

    # 2. ICA artifact removal
    ica = fit_ica(raw_filt, cfg.n_components, random_state=cfg.random_state)
    comp_scores = select_artifact_components(
        ica, raw_filt, eog_proxy=cfg.eog_proxy,
        eog_corr_thresh=cfg.eog_corr_thresh, muscle_thresh=cfg.muscle_thresh,
    )
    raw_clean = apply_ica(raw_filt, ica, comp_scores["excluded_components"])
    provenance["ica"] = comp_scores
    provenance["ica_n_components_fitted"] = int(ica.n_components_)

    # 3. Per-channel Z-score
    raw_z, norm_stats = zscore_normalize(raw_clean)
    provenance["normalization"] = norm_stats

    # 4. Resample to common target
    raw_out, was_resampled = resample_to_target(raw_z, cfg.target_sfreq)
    provenance["resampled"] = was_resampled
    provenance["output_sfreq"] = raw_out.info["sfreq"]

    provenance["fif_timestamps"] = sanitize_fif_timestamps(raw_out)

    provenance["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    return raw_out, to_jsonable(provenance)


if __name__ == "__main__":
    """Pilot runner: process every recording in each chbXX folder where
    XX is divisible by 4 (chb04, chb08, chb12, chb16, chb20).

    NOTE: this bypasses the shared Stage-0 loader (channel standardization,
    QC, special-montage routing) because Stage 0 is not implemented yet.
    Every provenance dict is tagged accordingly so benchmark results from
    this runner are not mistaken for protocol-valid runs later.
    """
    import json
    import re
    import traceback
    from pathlib import Path

    import mne

    DATA_ROOT = Path("/run/media/rangerofdanger/Files/Data Speaks/data/raw/physionet.org/")
    OUT_DIR = Path("/run/media/rangerofdanger/Files/Data Speaks/data/processed/pipeline_A")
    PROV_DIR = Path("/run/media/rangerofdanger/Files/Data Speaks/results/benchmarks/pipeline_A")
    PATIENT_DIVISOR = 4

    # --- Quick-test override -------------------------------------------------
    # ACTIVE  (line below has no '#'):  runs ONLY chb01  -> smoke test
    # COMMENTED OUT ('#' at line start): runs chb04, 08, 12, 16, 20 -> sweep
    OVERRIDE_PATIENT = "chb01"
    # -------------------------------------------------------------------------

    def selected_patient_dirs(root: Path) -> list[Path]:
        """All chbXX folders with XX % PATIENT_DIVISOR == 0."""
        dirs = []
        for d in sorted(root.iterdir()):
            m = re.fullmatch(r"chb(\d{2})", d.name)
            if d.is_dir() and m and int(m.group(1)) % PATIENT_DIVISOR == 0:
                dirs.append(d)
        return dirs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROV_DIR.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "pipeline_id": PIPELINE_ID,
        "patients": [],
        "successes": [],
        "failures": [],
        "provenance": {},
    }

    def write_run_summary():
        with open(PROV_DIR / "run_summary.json", "w") as fh:
            json.dump(run_summary, fh, indent=2)

    # .get() so commenting out the OVERRIDE_PATIENT line above safely
    # falls back to the full sweep instead of raising NameError.
    override = globals().get("OVERRIDE_PATIENT")
    patient_dirs = ([DATA_ROOT / override]
                    if override
                    else selected_patient_dirs(DATA_ROOT))
    missing = [str(d) for d in patient_dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Patient folder(s) not found under {DATA_ROOT}: {missing}. "
            "Check OVERRIDE_PATIENT / DATA_ROOT."
        )

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        run_summary["patients"].append(patient_id)
        print(f"=== {patient_id} ===")

        for edf_path in sorted(patient_dir.glob("*.edf")):
            recording_id = edf_path.stem
            try:
                # Direct EDF read — Stage 0 bypass, flagged in provenance.
                raw_std = mne.io.read_raw_edf(edf_path, preload=True,
                                              verbose=False)
                raw_out, prov = run_pipeline_a(raw_std,
                                               recording_id=recording_id)
                prov["stage0_bypassed"] = True  # not yet protocol-valid

                raw_out.save(OUT_DIR / f"{recording_id}_raw.fif",
                             overwrite=True, verbose=False)
                run_summary["successes"].append(recording_id)
                run_summary["provenance"][recording_id] = prov

                print(f"  OK   {recording_id} "
                      f"({prov['runtime_seconds']}s, "
                      f"ICs excluded: {prov['ica']['excluded_components']})")
            except Exception:
                run_summary["failures"].append(
                    {"recording_id": recording_id,
                     "traceback": traceback.format_exc()}
                )
                print(f"  FAIL {recording_id} — see run_summary.json")

            write_run_summary()