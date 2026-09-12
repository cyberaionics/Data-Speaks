from pathlib import Path
import re
import numpy as np

ICTAL_OVERLAP = 0.5
BOUNDARY_MARGIN_SEC = 30.0


def parse_seizure_file(edf_path):
    edf_path = Path(edf_path)
    text_annotation = edf_path.with_suffix(".seizure")
    if text_annotation.exists():
        out = []
        for line in text_annotation.read_text().splitlines():
            toks = line.strip().split()
            if len(toks) >= 2:
                try:
                    out.append((float(toks[0]), float(toks[1])))
                except ValueError:
                    continue
        return out

    summary_path = edf_path.parent / f"{edf_path.parent.name}-summary.txt"
    if not summary_path.exists():
        return []

    match = re.search(
        rf"File Name: {re.escape(edf_path.name)}\s+(.*?)(?=\nFile Name:|\Z)",
        summary_path.read_text(),
        flags=re.DOTALL,
    )
    if not match:
        return []

    starts = re.findall(r"Seizure Start Time:\s*([0-9.]+)", match.group(1))
    ends = re.findall(r"Seizure End Time:\s*([0-9.]+)", match.group(1))
    return [(float(start), float(end)) for start, end in zip(starts, ends)]


def label_windows(starts, epoch_sec, W, seizures, margin = BOUNDARY_MARGIN_SEC):
    win_len = epoch_sec * W
    labels = np.full(len(starts), -1, dtype = np.int8)
    for i, t0 in enumerate(starts):
        t1 = t0 + win_len
        max_overlap = 0.0
        min_gap = np.inf
        for s0, s1 in seizures:
            overlap = max(0.0, min(t1, s1) - max(t0, s0))
            max_overlap = max(max_overlap, overlap)
            if t1 <= s0:
                min_gap = min(min_gap, s0 - t1)
            elif t0 >= s1:
                min_gap = min(min_gap, t0 - s1)
        if max_overlap / win_len >= ICTAL_OVERLAP:
            labels[i] = 1
        elif max_overlap == 0.0 and min_gap >= margin:
            labels[i] = 0
    return labels