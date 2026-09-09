from mne.io import BaseRaw

DEFAULT_L_FREQ = 1.0
DEFAULT_H_FREQ = 40.0

def fir_bandpass(raw: BaseRaw, l_freq: float = DEFAULT_L_FREQ, h_freq: float = DEFAULT_H_FREQ, n_jobs:int = 1):
    raw_filt = raw.copy()
    raw_filt.load_data()
    raw_filt.filter(l_freq = l_freq, h_freq = h_freq, method = "fir", fir_design = "firwin", n_jobs = n_jobs, verbose = False)
    return raw_filt