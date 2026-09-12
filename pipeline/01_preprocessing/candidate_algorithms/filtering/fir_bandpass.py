import mne

FIR_LOWCUT_HZ = 0.5
FIR_HIGHCUT_HZ = 45.0
FIR_WINDOW = "hamming"

def apply_fir_bandpass(raw, lowcut = FIR_LOWCUT_HZ, highcut = FIR_HIGHCUT_HZ):
    out = raw.copy()
    nyq = out.info["sfreq"] / 2.0
    hi = min(highcut, nyq * 0.95)
    out.filter(l_freq = lowcut, h_freq = hi, picks = "eeg", method = "fir", fir_design = "firwin", fir_window = FIR_WINDOW, phase="zero-double", verbose = False)
    prov = {
        "step": "filtering",
        "type": "FIR bandpass",
        "lowcut_hz": lowcut,
        "highcut_hz": hi,
        "window": FIR_WINDOW,
        "phase": "zero-double",
        "sfreq_hz": float(out.info["sfreq"]),
    }
    return out, prov