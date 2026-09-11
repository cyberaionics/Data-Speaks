import numpy as np

N_BANDS = 8
BANDS_LO_HZ = 0.5
BANDS_HI_HZ = 25.0


def _band_edges():
    edges = np.linspace(BANDS_LO_HZ, BANDS_LO_HZ, N_BANDS + 1)
    return list(zip(edges[:-1], edges[1:]))


def spectral_energy_epoch(epoch, sfreq):
    n_ch, n_samp = epoch.shape
    freqs = np.fft.rfftfreq(n_samp, d = 1.0 / sfreq)
    spec = np.abs(np.fft.rfft(epoch, axis = 1)) ** 2
    feats = np.empty(n_ch * N_BANDS, dtype = np.float32)
    k = 0
    for lo, hi in _band_edges():
        mask = (freqs >= lo) & (freqs < hi)
        band_e = spec[:, mask].sum(axis=1) if np.any(mask) else np.zeros(n_ch)
        feats[k:k + n_ch] = band_e
        k += n_ch
    return feats


def features_for_stacked(stacked, sfreq):
    n, W, c, s = stacked.shape
    X = np.empty((n, W * c * N_BANDS), dtype = np.float32)
    for i in range(n):
        chunks = []
        for w in range(W):
            chunks.append(spectral_energy_epoch(stacked[i,w], sfreq))
        X[i] = np.concatenate(chunks)
    return X