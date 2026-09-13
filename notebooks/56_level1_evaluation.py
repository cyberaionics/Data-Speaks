import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/physionet.org/chb02/chb02_01.edf"
SUMMARY_PATH = ROOT / "data/raw/physionet.org/chb02/chb02-summary.txt"
RESULTS_DIR = ROOT / "results/benchmarks"
SEED = 42
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage0 = load_module("stage0_level1", ROOT / "pipeline/preprocessing_common/stage0.py")
pipelines = {
    "A": load_module("pipeline_a_level1", ROOT / "pipeline/01_preprocessing/candidate_algorithms/pipeline_a.py"),
    "B": load_module("pipeline_b_level1", ROOT / "pipeline/01_preprocessing/candidate_algorithms/pipeline_b.py"),
    "C": load_module("pipeline_c_level1", ROOT / "pipeline/01_preprocessing/candidate_algorithms/pipeline_c.py"),
    "D": load_module("pipeline_d_level1", ROOT / "pipeline/01_preprocessing/candidate_algorithms/pipeline_d.py"),
}


def band_powers(data, sfreq):
    frequencies, powers = welch(
        data, fs=sfreq, axis=1, nperseg=min(2048, data.shape[1])
    )
    return {
        name: powers[:, (frequencies >= low) & (frequencies < high)].mean(axis=1)
        for name, (low, high) in BANDS.items()
    }


def relative_band_powers(data, sfreq):
    powers = band_powers(data, sfreq)
    total = sum(powers.values())
    return {name: values / total for name, values in powers.items()}


def standardized(data):
    data = np.asarray(data)
    return (data - data.mean()) / data.std()


def main():
    raw, _ = stage0.load_stage0(RAW_PATH, summary_file=SUMMARY_PATH)
    raw.crop(tmin=0.0, tmax=60.0)
    clean = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=clean.shape) * clean.std(axis=1, keepdims=True) * 0.05
    noisy = clean + noise
    input_snr = 10.0 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2))
    raw_noisy = mne.io.RawArray(noisy, raw.info.copy(), verbose=False)
    results = []
    for name, module in pipelines.items():
        config = getattr(module, f"Pipeline{name}Config")()
        started = time.perf_counter()
        processed_clean, _ = getattr(module, f"run_pipeline_{name.lower()}")(raw.copy(), config=config, recording_id="chb02_01")
        clean_runtime = time.perf_counter() - started
        started = time.perf_counter()
        processed_noisy, _ = getattr(module, f"run_pipeline_{name.lower()}")(raw_noisy, config=config, recording_id="chb02_01_noisy")
        noisy_runtime = time.perf_counter() - started
        clean_output = processed_clean.get_data()
        noisy_output = processed_noisy.get_data()
        if clean_output.shape != noisy_output.shape:
            raise RuntimeError(f"Output shape mismatch in Pipeline {name}")
        output_noise = noisy_output - clean_output
        output_snr = 10.0 * np.log10(np.mean(clean_output ** 2) / np.mean(output_noise ** 2))
        raw_band = relative_band_powers(clean, sfreq)
        output_band = relative_band_powers(
            clean_output, float(processed_clean.info["sfreq"])
        )
        preservation = {
            band: float(np.mean(output_band[band] / raw_band[band]))
            for band in BANDS
        }
        correlation = float(
            np.mean(
                [
                    np.corrcoef(
                        standardized(clean[index]),
                        standardized(clean_output[index]),
                    )[0, 1]
                    for index in range(clean.shape[0])
                ]
            )
        )
        results.append(
            {
                "pipeline": name,
                "recording_id": "chb02_01",
                "frequency_bands_hz": BANDS,
                "psd_preservation_ratio": preservation,
                "snr_input_db": float(input_snr),
                "snr_output_db": float(output_snr),
                "snr_improvement_db": float(output_snr - input_snr),
                "channel_correlation": correlation,
                "runtime_sec_clean": clean_runtime,
                "runtime_sec_noisy": noisy_runtime,
                "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
                "seed": SEED,
                "noise_scale": 0.05,
            }
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "level1_evaluation.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    pd.DataFrame(results).to_csv(RESULTS_DIR / "level1_evaluation_summary.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()