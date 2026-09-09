from __future__ import annotations

from typing import Any

import mne


def apply_ica_artifact_removal(
    raw: mne.io.BaseRaw,
    n_components: int = 15,
    random_state: int = 42,
    eog_ch: str = "Fp1-F7",
    eog_threshold: float = 0.4,
    muscle_threshold: float = 0.3,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """
    Apply ICA-based artifact removal to FIR-filtered EEG.

    Pipeline A specification:
        - FastICA
        - up to 15 components, limited by data rank
        - random_state=42
        - Fp1-F7 used as an EOG proxy
        - muscle-component detection
        - return cleaned EEG and auditable provenance
    """

    if len(raw.ch_names) == 0:
        raise ValueError("Input recording contains no channels.")

    if n_components <= 0:
        raise ValueError("n_components must be greater than 0.")

    if not (0.0 < eog_threshold < 1.0):
        raise ValueError("eog_threshold must be between 0 and 1.")

    if not (0.0 < muscle_threshold < 1.0):
        raise ValueError("muscle_threshold must be between 0 and 1.")

    filtered_raw = raw.copy()

    # Verify the EOG proxy exists.
    if eog_ch not in filtered_raw.ch_names:
        raise ValueError(
            f"EOG proxy channel '{eog_ch}' was not found. "
            f"Available channels: {filtered_raw.ch_names}"
        )

    # Estimate the rank so ICA does not request more components
    # than the actual dimensionality of the EEG data.
    rank = mne.compute_rank(
        filtered_raw,
        rank="info",
        verbose=False,
    )

    eeg_rank = rank.get("eeg")

    if eeg_rank is None or eeg_rank < 1:
        raise ValueError(
            f"Could not determine a valid EEG rank: {eeg_rank}"
        )

    actual_n_components = min(
        n_components,
        int(eeg_rank),
        len(filtered_raw.ch_names),
    )

    ica = mne.preprocessing.ICA(
        n_components=actual_n_components,
        method="fastica",
        random_state=random_state,
        max_iter="auto",
    )

    ica.fit(
        filtered_raw,
        picks="eeg",
        verbose=False,
    )

    eog_indices: list[int] = []
    eog_scores = None

    try:
        eog_indices, eog_scores = ica.find_bads_eog(
            filtered_raw,
            ch_name=eog_ch,
            threshold=eog_threshold,
	    measure="correlation",
            verbose=False,
        )
    except Exception:
        # EOG proxy detection is a heuristic. If it cannot be
        # calculated, retain an empty result and continue so the
        # provenance records the issue.
        eog_indices = []
        eog_scores = None

    muscle_indices: list[int] = []
    muscle_scores = None

    try:
        muscle_indices, muscle_scores = ica.find_bads_muscle(
            filtered_raw,
            threshold=muscle_threshold,
            verbose=False,
        )
    except Exception:
        muscle_indices = []
        muscle_scores = None

    exclude = sorted(
        set(
            int(index)
            for index in eog_indices + muscle_indices
        )
    )

    ica.exclude = exclude

    cleaned = filtered_raw.copy()

    ica.apply(
        cleaned,
        verbose=False,
    )

    provenance: dict[str, Any] = {
        "method": "FastICA",
        "requested_n_components": n_components,
        "actual_n_components": actual_n_components,
        "estimated_eeg_rank": int(eeg_rank),
        "random_state": random_state,
        "eog_proxy_channel": eog_ch,
        "eog_threshold": eog_threshold,
        "eog_indices": [int(i) for i in eog_indices],
        "eog_scores": (
            eog_scores.tolist()
            if eog_scores is not None
            else None
        ),
        "muscle_threshold": muscle_threshold,
        "muscle_indices": [int(i) for i in muscle_indices],
        "muscle_scores": (
            muscle_scores.tolist()
            if muscle_scores is not None
            else None
        ),
        "excluded_components": exclude,
    }

    return cleaned, provenance
