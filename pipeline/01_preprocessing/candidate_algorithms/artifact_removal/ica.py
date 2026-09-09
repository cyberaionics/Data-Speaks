import numpy as np
from mne.io import BaseRaw
from mne.preprocessing import ICA

DEFAULT_N_COMPONENTS = 15
DEFAULT_RANDOM_STATE = 42
DEFAULT_EOG_PROXY = "auto"  
DEFAULT_EOG_CORR = 0.4


def resolve_eog_proxy(raw: BaseRaw, preferred: str = DEFAULT_EOG_PROXY) -> str | None:
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
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def fit_ica(raw: BaseRaw, n_components: int = DEFAULT_N_COMPONENTS, method: str = "fastica", random_state: int = DEFAULT_RANDOM_STATE, max_iter: int = "auto") -> ICA:
    n_eeg = len(raw.ch_names)
    n_components = min(n_components, n_eeg - 1)
    ica = ICA(n_components=n_components, method=method, random_state=random_state, max_iter=max_iter, fit_params=dict(tol=1e-4))
    ica.fit(raw, picks="eeg", verbose=False)
    return ica


def select_artifact_components(
    ica: ICA, raw: BaseRaw, eog_proxy: str = DEFAULT_EOG_PROXY, eog_corr_thresh: float = DEFAULT_EOG_CORR, muscle_thresh: float = 0.3) -> dict:
    scores = {"eog_corr_thresh": eog_corr_thresh}
    proxy = resolve_eog_proxy(raw, eog_proxy)
    scores["eog_proxy_ch"] = proxy
    if proxy is not None:
        bads, eog_scores = _mne_returns_scores(ica.find_bads_eog(raw, ch_name=proxy,threshold=eog_corr_thresh, verbose=False))
        scores["eog_scores"] = (np.asarray(eog_scores).tolist() if eog_scores is not None else None)
        scores["eog_bads"] = list(bads)
    else:
        scores["eog_bads"] = []
        scores["eog_scores"] = None
        scores["eog_note"] = ("no frontal channel found; " "eye-artifact detection skipped")

    try:
        bads_m, muscle_scores = _mne_returns_scores(ica.find_bads_muscle(raw, threshold=muscle_thresh, verbose=False))
        scores["muscle_scores"] = (np.asarray(muscle_scores).tolist() if muscle_scores is not None else None)
        scores["muscle_bads"] = list(bads_m)
    except Exception as exc:
        scores["muscle_bads"] = []
        scores["muscle_scores"] = None
        scores["muscle_note"] = f"find_bads_muscle failed: {exc!r}"

    scores["excluded_components"] = sorted(set(scores["eog_bads"]) | set(scores["muscle_bads"]))
    return scores


def apply_ica(raw: BaseRaw, ica: ICA, exclude: list) -> BaseRaw:
    raw_clean = raw.copy()
    ica.exclude = list(exclude)
    ica.apply(raw_clean, verbose=False)
    return raw_clean