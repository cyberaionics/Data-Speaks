import mne

TARGET_SFREQ_HZ = 256.0

def resample_to_256(raw, target_sfreq = TARGET_SFREQ_HZ):
    out = raw.copy()
    original =  float(out.info["sfreq"])
    if abs(original - target_sfreq) < 1e-6:
        return out, {"step":"Resampling", "original_sfreq_hz":original, "target_sfreq_hz":target_sfreq, "resampled":False}
    out.resample(target_sfreq, verbose = "False")
    return out, {"step":"Resampling", "original_sfreq_hz":original, "target_sfreq_hz":target_sfreq, "resampled":True}