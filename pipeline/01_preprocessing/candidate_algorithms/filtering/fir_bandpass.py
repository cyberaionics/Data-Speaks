"""Pipeline A — substep 1: FIR band-pass filtering.

Rationale (documented per project rule #11):
- FIR (not IIR) chosen for controlled frequency response and linear phase
  (no phase distortion of seizure morphology — important for downstream
  time-domain features).
- MNE's default ``firwin`` design is zero-phase (non-causal, forward-backward
  filtered), which preserves waveform timing; zero-phase filters introduce
  pre-ringing, accepted here because CHB-MIT has no strict causality
  requirement (offline analysis).
- Passband 1-40 Hz: the 1 Hz high-pass removes electrode drift / DC offsets
  and slow baseline wander without attenuating seizure delta activity; the
  40 Hz low-pass removes line noise (CHB-MIT recorded in the US -> 60 Hz),
  muscle and electrical artifact above the EEG band.
- No separate notch filter: 60 Hz sits above the 40 Hz cutoff, so the
  band-pass already suppresses it. A dedicated notch would only be needed
  if the low-pass edge were raised above 60 Hz in a later iteration.
"""

from mne.io import BaseRaw

DEFAULT_L_FREQ = 1.0
DEFAULT_H_FREQ = 40.0


def fir_bandpass(
    raw: BaseRaw,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    n_jobs: int = 1,
) -> BaseRaw:
    """Apply zero-phase FIR band-pass filtering to an EEG Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Standardized EEG input from Common Stage 0 (EEG channels only,
        annotations parsed, preloaded or loadable).
    l_freq, h_freq : float
        Band edges in Hz. Defaults 1.0-40.0 Hz.
    n_jobs : int
        Number of parallel jobs for MNE filtering.

    Returns
    -------
    raw_filt : mne.io.BaseRaw
        A **copy** — the input Raw is never modified in place
        (raw data is immutable, project rule #1).
    """
    raw_filt = raw.copy()
    raw_filt.load_data()
    raw_filt.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",  # MNE default design, zero-phase
        n_jobs=n_jobs,
        verbose=False,
    )
    return raw_filt