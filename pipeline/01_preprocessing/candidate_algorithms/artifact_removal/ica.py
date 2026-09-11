import mne

ICA_METHOD = "fastica"
ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = 500
EOG_THRESHOLD = 3.0
MUSCLE_THRESHOLD = 0.9

def remove_artifacts_ica(raw, n_components = ICA_N_COMPONENTS):
    out = raw.copy()
    n_comp = min(n_components, len(out.ch_names) - 1)
    if n_comp < 2:
        return out, {"step":"artifact_removal", "type":"ICA", "skipped":True, "reason":"Too few channels"}
    ica = mne.preprocessing.ICA(n_components = n_comp, method = ICA_METHOD, random_state = ICA_RANDOM_STATE, max_iter = ICA_MAX_ITER, verbose = False)
    
    try:
        ica.fit(out, verbose = False)
    except Exception as e:
        return out, {"step": "artifact_removal", "type":"ICA", "skipped":True, "reason":f"Step Failed: {e}"}
    
    bad = set()

    for ch in out.ch_names:
        if ch.upper().startswith(("FP1","FP2")):
            try:
                inds, _ = ica.find_bads_eog(out, ch_name = ch, threshold = EOG_THRESHOLD, verbose = False)
                bad.update(inds)
            except:
                pass

    try:
        m_inds, _ = ica.find_bads_muscle(out, threshold = MUSCLE_THRESHOLD, verbose = False)
        bad.update(m_inds)
    except Exception:
        pass

    bad_list = sorted(bad)[:n_comp // 2]
    try:
        ica.apply(out, exclude = bad_list, verbose = False)
    except Exception:
        pass

    prov = {"step":"artifact_removal","type":"ICA","method":ICA_METHOD, "n_components_fit":n_comp, "n_removed":len(bad_list), "removed":bad_list, "eog_threshold":EOG_THRESHOLD, "muscle_threshold":MUSCLE_THRESHOLD, "random_state":ICA_RANDOM_STATE}
    return out, prov