TARGET_SFREQ = 256.0


def resample_to_target(raw, target_sfreq: float = TARGET_SFREQ):
    current = raw.info["sfreq"]
    if abs(current - target_sfreq) < 1e-6:
        return raw.copy(), False
    raw_res = raw.copy()
    raw_res.resample(target_sfreq, npad="auto", verbose=False)
    return raw_res, True