import numpy as np
from mne.preprocessing import ICA
from mne.io import BaseRaw

DEFAULT_N_COMPONENTS = 15
DEFAULT_RANDOM_STATE = 42
DEFAULT_EOG_PROXY = "Fp1"
DEFAULT_EOG_CORR = 0.4


def mne_return_scores(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def fit_ica(raw: BaseRaw, n_components: int = DEFAULT_N_COMPONENTS, method: str = "fastica", random_state: int = DEFAULT_RANDOM_STATE, max_iter: int = "auto") -> ICA:
    n_eeg = len(raw.ch_names)
    n_components = min(n_components, n_eeg-1)
    ica = ICA(n_components = n_components, method = method, random_state = random_state, max_iter = max_iter, fit_params = dict(tol=1e-4))
    ica.fit(raw, pick = 'eeg', verbose = False)
    return ica


def select_artifact_components(ica:ICA, raw: BaseRaw, eog_proxy: str = DEFAULT_EOG_PROXY, eog_proxy_thresh: float = DEFAULT_EOG_CORR, muscle_thresh: float = 0.3) -> dict:
    scores = {"eog_proxy_ch": eog_proxy, "eog_corr_thres": eog_proxy_thresh}
    bads, eog_scores = _mne_returns_scores(ica.find_bads_eog(raw, ch_name = eog_proxy, threshold = eog_proxy_thresh, verbose = False))
    scores["eog_scores"] = (np.asarray(eog_scores).tolist if eog_scores is not None else None)
    scores["eog_bads"] = list(bads)
    bads_m, muscle_scores = _mne_returns_scores(ica.find_bads_muscle(raw, threshold = muscle_thresh, verbose = False))
    scores["muscle_scores"] = (np.asarray(muscle_scores).tolist() if muscle_scores is not None else None)
    scores["muscle_bads"] = list(bads_m)
    scores["excluded_components"] = sorted(set(scores["eog_bads"]) | set(scores["muscle_bads"]))
    return scores

def apply_ica(raw: BaseRaw, ica: ICA, exclude: list) -> BaseRaw:
    raw_clean = raw.copy()
    ica.exclude = list(exclude)
    ica.apply(raw_clean, verbose = False)
    return raw_clean