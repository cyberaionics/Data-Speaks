import mne

NON_EEG_PATTERNS = ("ECG", "EKG", "VNS", "SCL", "PHOTIC", "STI", "X1", "X2", "PULSE", "RESP", "TEMP", "EMG", "EOG")

STANDARD_18 = ("FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ")

def _norm(name: str) -> str:
    return name.upper().replace(" ", "")

def load_and_select_18(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload = True, verbose = False)
    original = list(raw.ch_names)
    eeg_only = [c for c in original if not any(p in c.upper() for p in NON_EEG_PATTERNS)]
    nmap = {_norm(c): c for c in eeg_only}
    keep, missing = [], []
    for target in STANDARD_18:
        key = _norm(target)
        if key in nmap:
            keep.append(nmap[key])
        else:
            missing.append(key)
    if len(keep) < 10:
        raise ValueError(f"Only {len(keep)}/18 standard channels in {edf_path}; missing: {missing}")
    raw.pick_channels(keep)
    info = {
        "original_channel_count": len(original),
        "eeg_channel_count": len(eeg_only),
        "retained_channel_count": len(keep),
        "retained_channels": keep,
        "missing_standard_channels": missing,
        "original_sfreq_hz": float(raw.info["sfreq"]),
    }
    return raw, info