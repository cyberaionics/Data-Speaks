import numpy as np 

EPOCH_SEC = 2.0
STACK_W = 3
STACK_OVERLAP = 0.5

def segment_epochs(data, sfreq, epoch_sec = EPOCH_SEC):
    n = int(epoch_sec * sfreq)
    n_epochs = data.shape[1] // n
    epochs = np.empty((n_epochs, data.shape[0], n), dtype = np.float32)
    starts = np.empty(n_epochs, dtype = np.float32)
    for i in range(n_epochs):
        s = i * n
        epochs[i] = data[:, s:s + n]
        starts[i] = s / sfreq
    return epochs, starts


def stack_epochs(epochs, starts, W = STACK_W, overlap = STACK_OVERLAP):
    step = max(1, int(round(W * (1 - overlap))))
    idx = list(range(0, len(epochs) - W + 1, step))
    if not idx:
        return (np.empty((0, W, epochs.shape[1], epochs.shape[2]), dtype=np.float32), np.empty(0))
    stacked = np.stack([epochs[i:i + W] for i in idx], axis=0).astype(np.float32)
    starts_out = np.array([starts[i] for i in idx], dtype = np.float32)
    return stacked, starts_out
    