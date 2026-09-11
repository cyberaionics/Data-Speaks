import numpy as np
from frequency_domain.band_energy import features_for_stacked as spectral
from time_domain.hjorth import hjorth_for_stacked
from statistical.basic_stats import stats_for_stacked

FEATURE_SETS = {
    "spectral" : lambda st, sf: spectral(st, sf),
    "hjorth" : lambda st, sf: hjorth_for_stacked(st, sf),
    "stats" : lambda st, sf: stats_for_stacked(st, sf)
}


def extract(stacked, sfreq, sets = ("spectral",)):
    parts = [FEATURE_SETS[s](stacked, sfreq) for s in sets]
    return np.hstack(parts)