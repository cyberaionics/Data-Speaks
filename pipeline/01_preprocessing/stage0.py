from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

DATASET_DIR = Path(r"C:\Users\Kavya\Data-Speaks\data\CHB-MIT")

CORE_17 = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "P8-O2",
]

NON_EEG_PATTERNS = [
    r"^ECG",
    r"^EKG",
    r"^VNS",
    r"^EMG",
    r"^LOC",
    r"^ROC",
    r"^LUE",
    r"^--",
]

SPECIAL_MONTAGE_PATTERN = re.compile(
    r"-CS2$",
    re.IGNORECASE,
)

WINDOW_SEC = 4.0
OVERLAP = 0.50

@dataclass
class QCFlags:
    flat: bool = False
    high_amplitude: bool = False
    high_variance: bool = False
    high_freq: bool = False
    clipped: bool = False
    invalid: bool = False

    @property
    def any_flag(self) -> bool:
        return any(
            [
                self.flat,
                self.high_amplitude,
                self.high_variance,
                self.clipped,
                self.invalid,
            ]
        )


@dataclass
class ChannelQCSummary:
    channel: str
    total_windows: int
    flat_count: int = 0
    high_amplitude_count: int = 0
    high_variance_count: int = 0
    high_freq_count: int = 0
    clipped_count: int = 0
    invalid_count: int = 0

    @property
    def any_flag_count(self) -> int:
        return self._any_flag_windows

    _any_flag_windows: int = 0

    @property
    def pct_affected(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return 100.0 * self._any_flag_windows / self.total_windows

    def print_report(self) -> None:
        print(f"Channel: {self.channel}")
        print(f"Total windows:       {self.total_windows}")
        print(f"High-amplitude:      {self.high_amplitude_count:>5}")
        print(f"High-variance:       {self.high_variance_count:>5}")
        print(f"Flat:                {self.flat_count:>5}")
        print(f"Clipped:             {self.clipped_count:>5}")
        print(f"High-freq:           {self.high_freq_count:>5}")
        print(f"Invalid:             {self.invalid_count:>5}")
        print(f"\n\u2192 {self.pct_affected:.2f}% windows affected\n")


@dataclass
class SeizureInterval:
    start_sec: float
    end_sec: float


@dataclass
class Stage0Result:
    raw: mne.io.BaseRaw

    patient_id: str
    recording_id: str

    duration_sec: float
    sampling_rate: float

    channels_kept: list[str]
    channels_dropped: list[str]

    missing_core_channels: list[str]
    standard_montage_complete: bool

    is_special_montage: bool
    candidate_ready: bool

    seizure_intervals: list[SeizureInterval]

    channel_qc: dict[str, ChannelQCSummary]

    integrity_warnings: list[str] = field(default_factory=list)

    def to_metadata_row(
        self,
        preprocessing_method: str = "Stage0",
        output_location: str = "not_saved_yet",
    ) -> dict:

        return {
            "patient_id": self.patient_id,
            "recording_id": self.recording_id,
            "duration_sec": self.duration_sec,
            "sampling_rate": self.sampling_rate,

            "n_channels": len(self.channels_kept),
            "channels": ";".join(self.channels_kept),

            "missing_core_channels": ";".join(
                self.missing_core_channels
            ),

            "standard_montage_complete":
                self.standard_montage_complete,

            "is_special_montage":
                self.is_special_montage,

            "candidate_ready":
                self.candidate_ready,

            "seizure_presence":
                len(self.seizure_intervals) > 0,

            "seizure_intervals": ";".join(
                f"{x.start_sec:.2f}-{x.end_sec:.2f}"
                for x in self.seizure_intervals
            ),

            "n_channels_with_qc_flags": sum(
                qc.any_flag_count > 0
                for qc in self.channel_qc.values()
            ),

            "preprocessing_method":
                preprocessing_method,

            "output_location":
                output_location,
        }


def load_and_check_integrity(
    edf_path: str | Path,
) -> tuple[mne.io.BaseRaw, list[str]]:

    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(
            f"EDF not found: {edf_path}"
        )

    warnings_list = []

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose="ERROR",
        )

    warnings_list.extend(
        str(w.message)
        for w in caught
    )

    sfreq = float(raw.info["sfreq"])

    if sfreq <= 0:
        raise ValueError(
            f"Invalid sampling frequency: {sfreq}"
        )

    if raw.n_times == 0:
        raise ValueError(
            f"EDF contains zero samples: {edf_path}"
        )

    data = raw.get_data()

    if np.isnan(data).any():
        warnings_list.append(
            "NaN values detected"
        )

    if np.isinf(data).any():
        warnings_list.append(
            "Inf values detected"
        )

    if not np.isclose(sfreq, 256.0):
        warnings_list.append(
            f"Sampling rate is {sfreq} Hz, "
            f"expected 256 Hz"
        )

    return raw, warnings_list


def parse_seizure_annotations(
    summary_path: str | Path,
    recording_filename: str,
) -> list[SeizureInterval]:

    summary_path = Path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}"
        )

    text = summary_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    blocks = re.split(
        r"(?=File Name:\s*\S+)",
        text,
        flags=re.IGNORECASE,
    )

    target_block = None

    for block in blocks:

        match = re.search(
            r"File Name:\s*(\S+)",
            block,
            flags=re.IGNORECASE,
        )

        if match:

            filename = match.group(1).strip()

            if filename.lower() == recording_filename.lower():
                target_block = block
                break

    if target_block is None:
        raise ValueError(
            f"No entry for {recording_filename} "
            f"in {summary_path.name}"
        )

    starts = []
    ends = []

    for line in target_block.splitlines():

        line = line.strip()

        start_match = re.search(
            r"Seizure(?:\s+\d+)?\s+Start\s+Time\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*seconds",
            line,
            flags=re.IGNORECASE,
        )

        end_match = re.search(
            r"Seizure(?:\s+\d+)?\s+End\s+Time\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*seconds",
            line,
            flags=re.IGNORECASE,
        )

        if start_match:
            starts.append(
                float(start_match.group(1))
            )

        if end_match:
            ends.append(
                float(end_match.group(1))
            )

    if len(starts) != len(ends):

        raise ValueError(
            f"Mismatch in seizure annotations for "
            f"{recording_filename}: "
            f"{len(starts)} starts vs "
            f"{len(ends)} ends"
        )

    intervals = []

    for start, end in zip(starts, ends):

        if end <= start:
            raise ValueError(
                f"Invalid seizure interval in "
                f"{recording_filename}: "
                f"{start} -> {end}"
            )

        intervals.append(
            SeizureInterval(
                start_sec=start,
                end_sec=end,
            )
        )

    return intervals


def is_auxiliary_channel(channel_name: str) -> bool:
    return any(
        re.search(
            pattern,
            channel_name,
            flags=re.IGNORECASE,
        )
        for pattern in NON_EEG_PATTERNS
    )


def apply_channel_policy(
    raw: mne.io.BaseRaw,
) -> tuple[
    mne.io.BaseRaw,
    list[str],
    list[str],
    list[str],
    bool,
    bool,
]:

    all_channels = list(raw.ch_names)

    is_special_montage = any(
        SPECIAL_MONTAGE_PATTERN.search(ch)
        for ch in all_channels
    )

    auxiliary = [
        ch
        for ch in all_channels
        if is_auxiliary_channel(ch)
    ]

    if is_special_montage:

        channels_kept = [
            ch
            for ch in all_channels
            if ch not in auxiliary
        ]

        channels_dropped = auxiliary

        raw_kept = raw.copy().pick(
            channels_kept
        )

        return (
            raw_kept,
            channels_kept,
            channels_dropped,
            CORE_17.copy(),
            True,
            False,
        )

    missing = [
        ch
        for ch in CORE_17
        if ch not in all_channels
    ]

    channels_kept = [
        ch
        for ch in CORE_17
        if ch in all_channels
    ]

    channels_dropped = [
        ch
        for ch in all_channels
        if ch not in CORE_17
    ]

    raw_kept = raw.copy().pick(
        channels_kept
    )

    standard_complete = (
        len(missing) == 0
    )

    candidate_ready = (
        standard_complete
        and len(channels_kept) == len(CORE_17)
    )

    return (
        raw_kept,
        channels_kept,
        channels_dropped,
        missing,
        False,
        candidate_ready,
    )


def _check_window(
    sig: np.ndarray,
    sfreq: float,
    channel_median_var: float,
    flat_std_threshold: float,
    amplitude_robust_z_threshold: float,
    variance_ratio_threshold: float,
    high_freq_band: tuple[float, float],
    high_freq_power_ratio_threshold: float,
    clip_repeat_threshold: int,
) -> QCFlags:
    flags = QCFlags()

    if np.isnan(sig).any() or np.isinf(sig).any():
        flags.invalid = True
        return flags

    if np.std(sig) < flat_std_threshold:
        flags.flat = True

    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    if mad > 0:
        robust_z = np.abs(sig - med) / (1.4826 * mad)
        if np.any(robust_z > amplitude_robust_z_threshold):
            flags.high_amplitude = True

    window_var = np.var(sig)
    if channel_median_var > 0 and window_var > variance_ratio_threshold * channel_median_var:
        flags.high_variance = True

    values, counts = np.unique(sig, return_counts=True)
    max_idx = np.argmax(counts)
    if counts[max_idx] >= clip_repeat_threshold and values[max_idx] in (sig.max(), sig.min()):
        flags.clipped = True

    freqs, psd = welch(sig, fs=sfreq, nperseg=min(int(sfreq * 2), len(sig)))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    total_power = trapz(psd, freqs)
    band_mask = (freqs >= high_freq_band[0]) & (freqs <= high_freq_band[1])
    if band_mask.any() and total_power > 0:
        band_power = trapz(psd[band_mask], freqs[band_mask])
        if (band_power / total_power) > high_freq_power_ratio_threshold:
            flags.high_freq = True

    return flags


def run_quality_checks_windowed(
    raw,
    window_sec: float = 4.0,
    overlap: float = 0.50,
    flat_std_threshold: float = 1e-7,
    amplitude_robust_z_threshold: float = 30.0,
    variance_ratio_threshold: float = 5.0,
    high_freq_band: tuple[float, float] = (40.0, 70.0),
    high_freq_power_ratio_threshold: float = 0.30,
    clip_repeat_threshold: int = 20,
) -> dict[str, list[QCFlags]]:
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    ch_names = raw.ch_names

    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))

    all_windows_per_channel: dict[str, list[np.ndarray]] = {ch: [] for ch in ch_names}
    starts = list(range(0, data.shape[1] - window_samples + 1, step_samples))
    for start in starts:
        stop = start + window_samples
        for i, ch in enumerate(ch_names):
            all_windows_per_channel[ch].append(data[i, start:stop])

    channel_median_var = {
        ch: float(np.median([np.var(w) for w in windows]))
        for ch, windows in all_windows_per_channel.items()
    }

    result: dict[str, list[QCFlags]] = {ch: [] for ch in ch_names}
    for ch in ch_names:
        for window in all_windows_per_channel[ch]:
            flags = _check_window(
                window,
                sfreq,
                channel_median_var[ch],
                flat_std_threshold,
                amplitude_robust_z_threshold,
                variance_ratio_threshold,
                high_freq_band,
                high_freq_power_ratio_threshold,
                clip_repeat_threshold,
            )
            result[ch].append(flags)

    return result


def summarize_channel_qc(windowed_qc: dict[str, list[QCFlags]]) -> dict[str, ChannelQCSummary]:
    summaries = {}
    for ch, flags_list in windowed_qc.items():
        s = ChannelQCSummary(channel=ch, total_windows=len(flags_list))
        for f in flags_list:
            if f.flat:
                s.flat_count += 1
            if f.high_amplitude:
                s.high_amplitude_count += 1
            if f.high_variance:
                s.high_variance_count += 1
            if f.high_freq:
                s.high_freq_count += 1
            if f.clipped:
                s.clipped_count += 1
            if f.invalid:
                s.invalid_count += 1
            if f.any_flag:
                s._any_flag_windows += 1
        summaries[ch] = s
    return summaries


def run_stage0(
    edf_path: str | Path,
    summary_path: str | Path,
    patient_id: str,
    recording_id: str,
) -> Stage0Result:

    edf_path = Path(edf_path)

    raw, integrity_warnings = (
        load_and_check_integrity(
            edf_path
        )
    )

    seizure_intervals = (
        parse_seizure_annotations(
            summary_path,
            edf_path.name,
        )
    )

    (
        raw_kept,
        channels_kept,
        channels_dropped,
        missing_core_channels,
        is_special,
        candidate_ready,
    ) = apply_channel_policy(raw)

    windowed_qc = run_quality_checks_windowed(
        raw_kept
    )

    channel_qc = summarize_channel_qc(
        windowed_qc
    )

    standard_complete = (
        len(missing_core_channels) == 0
        and not is_special
    )

    return Stage0Result(

        raw=raw_kept,

        patient_id=patient_id,
        recording_id=recording_id,

        duration_sec=float(
            raw_kept.n_times
            / raw_kept.info["sfreq"]
        ),

        sampling_rate=float(
            raw_kept.info["sfreq"]
        ),

        channels_kept=channels_kept,
        channels_dropped=channels_dropped,

        missing_core_channels=
            missing_core_channels,

        standard_montage_complete=
            standard_complete,

        is_special_montage=is_special,

        candidate_ready=candidate_ready,

        seizure_intervals=
            seizure_intervals,

        channel_qc=channel_qc,

        integrity_warnings=
            integrity_warnings,
    )
