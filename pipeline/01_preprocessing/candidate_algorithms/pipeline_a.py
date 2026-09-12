import mne

from filtering.fir_bandpass import apply_fir_bandpass
from artifact_removal.ica import remove_artifacts_ica
from normalization.zscore import zscore_normalize_raw
from resampling.resample_256 import resample_to_256

PIPELINE_ID = "pipeline_A_fir_ica_zscore_256"

def run_pipeline(raw_std, recording_id = None, return_metadata = True):
    prov = {"pipeline_id": PIPELINE_ID, "recording_id":recording_id, "steps":[]}
    raw, p1 = apply_fir_bandpass(raw_std)
    prov["steps"].append(p1)

    raw, p2 = remove_artifacts_ica(raw)
    prov["steps"].append(p2)

    raw, stats = zscore_normalize_raw(raw)
    prov["steps"].append({"step":"normalization", "type":"per-channel z-score", "mean_per_channel":stats["mean"].tolist(), "std_per_channel":stats["std"].tolist()})

    raw, p4 = resample_to_256(raw)
    prov["steps"].append(p4)

    prov["summary"] = {
        "n_channels": len(raw.ch_names),
        "n_samples": int(raw.n_times),
        "sfreq_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "channel_names": list(raw.ch_names),
    }

    if return_metadata:
        return raw, prov
    return raw