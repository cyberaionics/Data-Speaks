"""Pipeline A — substep 4: resampling to the common target rate (256 Hz).

Documented decisions:
- CHB-MIT is natively recorded at 256 Hz, so for the standard benchmark
  this substep is a verification no-op. The function still implements full
  polyphase resampling with FIR anti-aliasing (MNE ``raw.resample``) for
  any recording whose native rate differs (e.g. the 3 special-montage
  files must NOT reach here unhandled — Stage 0 routes them separately).
- Anti-aliasing is mandatory before decimation: naive downsampling without
  a low-pass would fold high-frequency energy into the band of interest.
  MNE's resample applies a polyphase FIR low-pass at the Nyquist of the
  target rate automatically; we do not disable it.
- Resampling happens AFTER filtering and ICA so ICA operates at native
  resolution and any resample-induced transients are not fed into the
  unmixing (order fixed by the pipeline contract; do not reorder).
"""

TARGET_SFREQ = 256.0


def resample_to_target(raw, target_sfreq: float = TARGET_SFREQ):
    """Resample to ``target_sfreq`` Hz with polyphase anti-aliasing.

    Returns (resampled copy, was_resampled: bool). No-op if already at
    the target rate.
    """
    current = raw.info["sfreq"]
    if abs(current - target_sfreq) < 1e-6:
        return raw.copy(), False
    raw_res = raw.copy()
    raw_res.resample(target_sfreq, npad="auto", verbose=False)
    return raw_res, True