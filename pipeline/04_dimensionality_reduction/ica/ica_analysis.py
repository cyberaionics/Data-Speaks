import mne 
import numpy as np

def ica_variance_profile(data, sfreq, n_components = 20, random_state = 42):
    ch_names = [f"EEG{i:03d}" for i in range(data.shape[0])]
    info = mne.create_info(ch_names, sfreq, ch_types = "eeg")
    raw = mne.io.RawArray(data, info, verbose = False)
    ica = mne.preprocessing.ICA(n_components=min(n_components, data.shape[0] - 1), method = "fastica", random_state = random_state, verbose = False)
    ica.fit(raw, verbose = False)
    sources = ica.get_sources(raw).get_data()
    var = sources.var(axis = 1)
    return var / var.sum(), len(var)