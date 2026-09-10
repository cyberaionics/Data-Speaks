from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.signal import welch

SPECTRAL_REPRESENTATIVE_CHANNELS = ["FP1-F7", "FZ-CZ", "P8-O2"]

# EEG band definitions (Hz)
BANDS = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha":  (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

def _trapz(psd: np.ndarray, freqs: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(psd, freqs))


def _band_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return _trapz(psd[mask], freqs[mask])


def _channel_signal_stats(data_1d: np.ndarray) -> dict:
    """Per-channel signal statistics."""
    rms = float(np.sqrt(np.mean(data_1d ** 2)))
    return {
        "mean":   float(np.mean(data_1d)),
        "std":    float(np.std(data_1d)),
        "var":    float(np.var(data_1d)),
        "rms":    rms,
        "max_abs": float(np.max(np.abs(data_1d))),
        "peak_to_peak": float(np.max(data_1d) - np.min(data_1d)),
    }


def _overall_signal_stats(data: np.ndarray) -> dict:
    """Overall (all-channel flattened) signal statistics."""
    flat = data.ravel()
    return _channel_signal_stats(flat)


def _compute_psd_stats(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    channels: list[str] | None = None,
) -> dict:
    """
    Compute PSD-based spectral metrics for selected channels.

    Parameters
    ----------
    channels : list or None
        If None, use all channels.
    """
    if channels is None:
        channels = ch_names

    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    results = {}

    for ch in channels:
        if ch not in ch_idx:
            continue
        idx = ch_idx[ch]
        freqs, psd = welch(
            data[idx],
            fs=sfreq,
            nperseg=min(int(sfreq * 4), data.shape[1]),
        )
        total = _trapz(psd, freqs)
        entry = {"total_power": total}
        for band_name, (lo, hi) in BANDS.items():
            bp = _band_power(psd, freqs, lo, hi)
            entry[f"{band_name}_power"] = bp
            entry[f"{band_name}_rel"] = bp / total if total > 0 else 0.0
        results[ch] = entry

    return results

def compute_structural_metrics(
    raw_input,
    raw_output,
) -> dict:
    """Compare structural properties before and after Pipeline C."""
    data_in  = raw_input.get_data()
    data_out = raw_output.get_data()

    return {
        "input_duration_sec":    float(raw_input.times[-1]),
        "output_duration_sec":   float(raw_output.times[-1]),
        "input_n_channels":      len(raw_input.ch_names),
        "output_n_channels":     len(raw_output.ch_names),
        "input_sfreq":           float(raw_input.info["sfreq"]),
        "output_sfreq":          float(raw_output.info["sfreq"]),
        "input_nan_count":       int(np.isnan(data_in).sum()),
        "output_nan_count":      int(np.isnan(data_out).sum()),
        "input_inf_count":       int(np.isinf(data_in).sum()),
        "output_inf_count":      int(np.isinf(data_out).sum()),
        "channels_preserved":    raw_input.ch_names == raw_output.ch_names,
        "sfreq_preserved":       abs(raw_input.info["sfreq"] - raw_output.info["sfreq"]) < 1e-6,
        "duration_preserved":    abs(raw_input.times[-1] - raw_output.times[-1]) < 0.02,
    }

def format_runtime_metrics(timings: dict) -> dict:
    """Pass-through formatter for timing dict from run_pipeline_c."""
    return {k: round(v, 4) for k, v in timings.items()}

def compute_signal_stats(
    raw_stage0,
    raw_filtered=None,
    raw_asr=None,
    raw_ica=None,
    raw_final=None,
) -> dict:
    """
    Signal statistics at each Pipeline C stage.

    Returns per-channel and overall statistics at each available stage.
    """
    stages = {
        "stage0": raw_stage0,
        "after_butterworth": raw_filtered,
        "after_asr": raw_asr,
        "after_ica": raw_ica,
        "final": raw_final,
    }

    result = {}
    for stage_name, raw in stages.items():
        if raw is None:
            continue
        data = raw.get_data()
        ch_stats = {}
        for i, ch in enumerate(raw.ch_names):
            ch_stats[ch] = _channel_signal_stats(data[i])
        result[stage_name] = {
            "per_channel": ch_stats,
            "overall": _overall_signal_stats(data),
        }

    return result

def compute_spectral_metrics(
    raw_input,
    raw_output,
    channels: list[str] | None = None,
    all_channels: bool = False,
) -> dict:
    """
    Spectral power comparison before and after Pipeline C.

    Parameters
    ----------
    channels : list[str] | None
        Channels to analyse.  If None, uses SPECTRAL_REPRESENTATIVE_CHANNELS.
    all_channels : bool
        If True, compute for all 17 channels (use for single recording only).
    """
    if all_channels:
        channels = None  # compute_psd_stats will use all
    elif channels is None:
        channels = [
            ch for ch in SPECTRAL_REPRESENTATIVE_CHANNELS
            if ch in raw_input.ch_names
        ]

    sfreq_in  = float(raw_input.info["sfreq"])
    sfreq_out = float(raw_output.info["sfreq"])

    data_in  = raw_input.get_data()
    data_out = raw_output.get_data()

    psd_in  = _compute_psd_stats(data_in,  sfreq_in,  list(raw_input.ch_names),  channels)
    psd_out = _compute_psd_stats(data_out, sfreq_out, list(raw_output.ch_names), channels)

    return {
        "input_psd":  psd_in,
        "output_psd": psd_out,
        "channels_analysed": list(psd_in.keys()),
    }

def compute_qc_metrics(channel_qc: dict) -> dict:
    """
    Summarise Stage 0 QC window flag counts across all channels.

    channel_qc is the dict[str, ChannelQCSummary] from Stage0Result.
    """
    if not channel_qc:
        return {}

    totals = {
        "total_windows_all_channels": 0,
        "flat_windows": 0,
        "high_amplitude_windows": 0,
        "high_variance_windows": 0,
        "high_freq_windows": 0,
        "clipped_windows": 0,
        "invalid_windows": 0,
        "any_flag_windows": 0,
    }

    per_ch = {}
    for ch, qc in channel_qc.items():
        totals["total_windows_all_channels"] += qc.total_windows
        totals["flat_windows"]           += qc.flat_count
        totals["high_amplitude_windows"] += qc.high_amplitude_count
        totals["high_variance_windows"]  += qc.high_variance_count
        totals["high_freq_windows"]      += qc.high_freq_count
        totals["clipped_windows"]        += qc.clipped_count
        totals["invalid_windows"]        += qc.invalid_count
        totals["any_flag_windows"]       += qc.any_flag_count
        per_ch[ch] = {
            "total_windows":       qc.total_windows,
            "flat":                qc.flat_count,
            "high_amplitude":      qc.high_amplitude_count,
            "high_variance":       qc.high_variance_count,
            "high_freq":           qc.high_freq_count,
            "clipped":             qc.clipped_count,
            "invalid":             qc.invalid_count,
            "any_flag":            qc.any_flag_count,
            "pct_affected":        round(qc.pct_affected, 3),
        }

    n_total = totals["total_windows_all_channels"]
    totals["pct_any_flag"] = (
        round(100.0 * totals["any_flag_windows"] / n_total, 3)
        if n_total > 0 else 0.0
    )

    return {"totals": totals, "per_channel": per_ch}

def compute_ica_metrics(ica, raw) -> dict:
    """
    Report ICA component counts and EOG proxy info.
    """
    n_components_fit = ica.n_components_
    n_excluded = len(ica.exclude)
    pct_excluded = (
        100.0 * n_excluded / n_components_fit
        if n_components_fit > 0 else 0.0
    )

    eog_proxy_used = [
        ch for ch in ["FP1-F7", "FP2-F4"]
        if ch in raw.ch_names
    ]

    return {
        "n_ica_components_fit":      n_components_fit,
        "n_components_excluded":     n_excluded,
        "pct_components_excluded":   round(pct_excluded, 2),
        "excluded_indices":          list(ica.exclude),
        "eog_proxy_channels":        eog_proxy_used,
        "ica_method":                ica.method,
        "ica_variance_threshold":    ica.n_components,
    }

def compute_data_preservation_metrics(
    raw_stage0,
    raw_filtered=None,
    raw_asr=None,
    raw_ica=None,
    raw_final=None,
) -> dict:
    """
    Quantify how much the signal changed at each Pipeline C stage.

    Metrics per stage vs. Stage 0 input:
      - rms_diff           : RMS of the difference signal (all channels)
      - relative_rms_change: rms_diff / rms_input  (fraction)
    """
    data0 = raw_stage0.get_data()
    rms0  = float(np.sqrt(np.mean(data0 ** 2)))

    stages = {
        "after_butterworth": raw_filtered,
        "after_asr":         raw_asr,
        "after_ica":         raw_ica,
        "final":             raw_final,
    }

    result = {"input_rms": rms0}

    for stage_name, raw in stages.items():
        if raw is None:
            continue
        data = raw.get_data()
        # Trim to same length (resampling can add/remove 1 sample)
        min_len = min(data0.shape[1], data.shape[1])
        diff    = data0[:, :min_len] - data[:, :min_len]
        rms_diff = float(np.sqrt(np.mean(diff ** 2)))
        rel_change = rms_diff / rms0 if rms0 > 0 else 0.0
        result[stage_name] = {
            "output_rms":           float(np.sqrt(np.mean(data ** 2))),
            "rms_diff":             rms_diff,
            "relative_rms_change":  round(rel_change, 6),
        }

    return result

def build_csv_row(
    patient_id: str,
    recording_id: str,
    stage0_result,
    pipeline_c_outputs: dict,
    spectral_channels: list[str] | None = None,
) -> dict:
    """
    Build a single flat dict suitable for one CSV row per recording.

    Parameters
    ----------
    stage0_result : Stage0Result
    pipeline_c_outputs : dict returned by run_pipeline_c()
    spectral_channels : channels to use for spectral metrics
                        (default: SPECTRAL_REPRESENTATIVE_CHANNELS)
    """
    raw0   = stage0_result.raw
    calib  = pipeline_c_outputs["calibration"]
    ica    = pipeline_c_outputs["ica"]
    timings = pipeline_c_outputs["timings"]
    final  = pipeline_c_outputs["final"]

    # -- Structural --
    struct = compute_structural_metrics(raw0, final)

    # -- Signal stats (before / after only for CSV) --
    d0   = raw0.get_data()
    df   = final.get_data()
    rms0 = float(np.sqrt(np.mean(d0 ** 2)))
    rmsf = float(np.sqrt(np.mean(df ** 2)))
    rms_diff = float(np.sqrt(np.mean(
        (d0[:, :df.shape[1]] - df[:, :d0.shape[1]]) ** 2
    )))

    # -- Spectral (representative channels) --
    if spectral_channels is None:
        spectral_channels = [
            ch for ch in SPECTRAL_REPRESENTATIVE_CHANNELS
            if ch in raw0.ch_names
        ]
    spec = compute_spectral_metrics(raw0, final, channels=spectral_channels)

    # Flatten spectral into representative-channel averages
    def _avg_band(psd_dict: dict, band: str) -> float:
        vals = [v.get(f"{band}_power", 0.0) for v in psd_dict.values()]
        return float(np.mean(vals)) if vals else 0.0

    # -- Seizure info --
    seizure_intervals_str = ";".join(
        f"{iv.start_sec:.2f}-{iv.end_sec:.2f}"
        for iv in stage0_result.seizure_intervals
    )

    row = {
        "patient_id":           patient_id,
        "recording_id":         recording_id,
        "duration_sec":         stage0_result.duration_sec,
        "seizure_presence":     len(stage0_result.seizure_intervals) > 0,
        "seizure_intervals":    seizure_intervals_str,
        "sampling_rate":        stage0_result.sampling_rate,
        "n_channels":           len(stage0_result.channels_kept),

        # ASR calibration
        "calibration_start_sec": calib.start_sec,
        "calibration_stop_sec":  calib.stop_sec,
        "calibration_reason":    calib.reason,
        "asr_cutoff":            20.0,

        # ICA
        "n_ica_components":      ica.n_components_,
        "n_ica_components_removed": len(ica.exclude),
        "ica_eog_proxy_channels": ";".join(
            ch for ch in ["FP1-F7", "FP2-F4"]
            if ch in raw0.ch_names
        ),

        # Timings
        "butterworth_runtime_sec":    round(timings["butterworth_sec"], 4),
        "asr_runtime_sec":            round(timings["asr_sec"], 4),
        "ica_runtime_sec":            round(timings["ica_sec"], 4),
        "standardization_runtime_sec":round(timings["standardization_sec"], 4),
        "total_pipeline_c_runtime_sec":round(timings["total_pipeline_c_sec"], 4),

        # Integrity
        "output_nan_count":    struct["output_nan_count"],
        "output_inf_count":    struct["output_inf_count"],

        # Signal stats
        "input_rms":           round(rms0, 8),
        "output_rms":          round(rmsf, 8),
        "input_variance":      round(float(np.var(d0)), 12),
        "output_variance":     round(float(np.var(df)), 12),
        "rms_diff":            round(rms_diff, 8),
        "relative_rms_change": round(rms_diff / rms0 if rms0 > 0 else 0.0, 6),

        # Spectral — input representative-channel averages
        "input_total_power_avg":  round(_avg_band(spec["input_psd"], "total"), 12),
        "input_delta_power_avg":  round(_avg_band(spec["input_psd"], "delta"), 12),
        "input_theta_power_avg":  round(_avg_band(spec["input_psd"], "theta"), 12),
        "input_alpha_power_avg":  round(_avg_band(spec["input_psd"], "alpha"), 12),
        "input_beta_power_avg":   round(_avg_band(spec["input_psd"], "beta"),  12),
        "input_gamma_power_avg":  round(_avg_band(spec["input_psd"], "gamma"), 12),

        # Spectral — output representative-channel averages
        "output_total_power_avg": round(_avg_band(spec["output_psd"], "total"), 12),
        "output_delta_power_avg": round(_avg_band(spec["output_psd"], "delta"), 12),
        "output_theta_power_avg": round(_avg_band(spec["output_psd"], "theta"), 12),
        "output_alpha_power_avg": round(_avg_band(spec["output_psd"], "alpha"), 12),
        "output_beta_power_avg":  round(_avg_band(spec["output_psd"], "beta"),  12),
        "output_gamma_power_avg": round(_avg_band(spec["output_psd"], "gamma"), 12),
    }

    return row
