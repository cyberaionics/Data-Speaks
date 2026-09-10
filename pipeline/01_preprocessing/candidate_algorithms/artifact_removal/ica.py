"""Pipeline A — substep 2: ICA artifact removal.

Rationale:
- ICA is among the most widely used multichannel EEG artifact-removal
  approaches (established BSS baseline; appropriate for the conservative
  reference pipeline A).
- FastICA is the standard, well-characterized ICA algorithm.

IMPORTANT — component labeling without reference channels:
- Common Stage 0 removes ECG/VNS and auxiliary channels, so the classic
  "correlate ICs with EOG/ECG channel" recipe must be approximated:
  1. EYE PROXY: `find_bads_eog` with ``ch_name="Fp1"`` — frontal channel
     picks up blink/eye-movement field topography. This is a *proxy*, not a
     true EOG measurement.
  2. MUSCLE: `find_bads_muscle` — spectral flat high-frequency signature.
- Both are heuristics. All correlation/spectral scores are written to the
  provenance dict so Level-1 evaluation can audit exactly which components
  were rejected and why (no undocumented defaults, project rule #11).
- If both heuristics fire on few components, prefer rechecking scores over
  aggressive rejection: over-cleaning risks removing seizure-relevant
  structure (project rule #9).
"""

import numpy as np
from mne.io import BaseRaw
from mne.preprocessing import ICA

DEFAULT_N_COMPONENTS = 15
DEFAULT_RANDOM_STATE = 42
DEFAULT_EOG_PROXY = "auto"   # resolved against actual channel names, see below
DEFAULT_EOG_CORR = 0.4


def resolve_eog_proxy(raw: BaseRaw, preferred: str = DEFAULT_EOG_PROXY) -> str | None:
    """Find a frontal channel to use as an eye-movement proxy.

    Why this exists: CHB-MIT channels are bipolar pairs with UPPERCASE
    names (e.g. "FP1-F7", "FP2-F8"), so the textbook proxy "Fp1" never
    matches literally and find_bads_eog raises ValueError. Resolution:
      1. exact (case-insensitive) match if `preferred` is a real name
      2. first channel containing "FP1" (case-insensitive)
      3. first channel containing "FP2", then "FP"
      4. None -> eye detection is SKIPPED (recorded in scores), never fatal
    """
    ch_names = list(raw.ch_names)
    if preferred and preferred.lower() != "auto":
        for ch in ch_names:
            if ch.lower() == preferred.lower():
                return ch
    for key in ("fp1", "fp2", "fp"):
        for ch in ch_names:
            if key in ch.lower():
                return ch
    return None


def _mne_returns_scores(result):
    """MNE versions differ: some find_bads_* return (bads, scores)."""
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def _has_sensor_positions(raw: BaseRaw) -> bool:
    locs = np.array([ch["loc"][:3] for ch in raw.info["chs"]], dtype=float)
    if locs.size == 0 or not np.isfinite(locs).any():
        return False
    return not np.allclose(locs, 0.0)


def fit_ica(
    raw: BaseRaw,
    n_components: int = DEFAULT_N_COMPONENTS,
    method: str = "fastica",
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = "auto",
) -> ICA:
    """Fit ICA on the (already band-pass filtered) standardized EEG.

    n_components is clamped to rank-1 so ICA cannot request more sources
    than the channel rank allows.
    """
    n_eeg = len(raw.ch_names)
    n_components = min(n_components, n_eeg - 1)
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=max_iter,
        fit_params=dict(tol=1e-4),  # documented: FastICA convergence tol
    )
    ica.fit(raw, picks="eeg", verbose=False)
    return ica


def select_artifact_components(
    ica: ICA,
    raw: BaseRaw,
    eog_proxy: str = DEFAULT_EOG_PROXY,
    eog_corr_thresh: float = DEFAULT_EOG_CORR,
    muscle_thresh: float = 0.3,
) -> dict:
    """Heuristically label artifact components. Returns a scores dict.

    Combines the frontal eye proxy and spectral muscle detector. Every
    score is retained for auditability.
    """
    scores = {"eog_corr_thresh": eog_corr_thresh}

    # --- eye-movement proxy (never fatal) ---
    proxy = resolve_eog_proxy(raw, eog_proxy)
    scores["eog_proxy_ch"] = proxy
    if proxy is not None:
        # MNE's default measure is z-score (threshold ~3). Pipeline A documents
        # eog_corr_thresh as a Pearson |r| cutoff, so we must request
        # measure="correlation" or almost every IC is marked as EOG.
        bads, eog_scores = _mne_returns_scores(
            ica.find_bads_eog(
                raw, ch_name=proxy, threshold=eog_corr_thresh,
                measure="correlation", verbose=False,
            )
        )
        scores["eog_scores"] = (
            np.asarray(eog_scores).tolist() if eog_scores is not None else None
        )
        scores["eog_bads"] = [int(x) for x in bads]
    else:
        scores["eog_bads"] = []
        scores["eog_scores"] = None
        scores["eog_note"] = ("no frontal channel found; "
                              "eye-artifact detection skipped")

    # --- muscle detector (never fatal) ---
    # CHB-MIT bipolar channels have no 3D locations. find_bads_muscle then
    # uses only the spectral-slope criterion, which over-rejects on this
    # dataset. Skip exclusion (still recorded) rather than strip most ICs.
    if not _has_sensor_positions(raw):
        scores["muscle_bads"] = []
        scores["muscle_scores"] = None
        scores["muscle_note"] = (
            "no sensor positions; muscle detection skipped "
            "(slope-only criterion would over-reject)"
        )
    else:
        try:
            bads_m, muscle_scores = _mne_returns_scores(
                ica.find_bads_muscle(raw, threshold=muscle_thresh, verbose=False)
            )
            scores["muscle_scores"] = (
                np.asarray(muscle_scores).tolist()
                if muscle_scores is not None
                else None
            )
            scores["muscle_bads"] = [int(x) for x in bads_m]
        except Exception as exc:  # logged, not fatal — pilot robustness
            scores["muscle_bads"] = []
            scores["muscle_scores"] = None
            scores["muscle_note"] = f"find_bads_muscle failed: {exc!r}"

    scores["excluded_components"] = sorted(
        {int(x) for x in scores["eog_bads"]}
        | {int(x) for x in scores["muscle_bads"]}
    )
    return scores


def apply_ica(raw: BaseRaw, ica: ICA, exclude: list) -> BaseRaw:
    """Reconstruct the signal with artifact components removed.

    Returns a copy; the input is never modified in place.
    """
    raw_clean = raw.copy()
    ica.exclude = list(exclude)
    ica.apply(raw_clean, verbose=False)
    return raw_clean