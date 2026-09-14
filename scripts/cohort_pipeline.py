"""
CHB-MIT Full Cohort Preprocessing, Feature Extraction & Modeling Engine (Approach 4)
----------------------------------------------------------------------------------
Processes all populated patients with clinical seizures:
1. Standard 18 Bipolar Montage Extraction
2. Chebyshev Type II Bandpass Filter (0.5 - 45 Hz)
3. Decimation (256 Hz -> 128 Hz)
4. Robust Statistical Artifact Screening
5. 126-Dimensional Feature Extraction (5 Frequency Bands + Variance + Kurtosis per channel)
6. Cohort Dataset Serialization (X_cohort.npy, y_cohort.npy, metadata)
7. Machine Learning Model Training (Random Forest with Balanced Weights)
8. Clinical Metrics Evaluation (Event Sensitivity, Detection Latency, False Alarms/24h)
9. Cohort-Wide EDA Visualizations
"""

import os
import re
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import mne

warnings.filterwarnings('ignore')

STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Gamma': (30.0, 45.0)
}

DATA_ROOT = Path(os.getenv('CHBMIT_DIR', Path(r'C:\Users\avani\chbmit')))
PROJECT_DIR = Path(r'C:\Users\avani\.gemini\antigravity\scratch\chbmit_pipeline_approach4')
OUTPUT_DIR = PROJECT_DIR / 'output'
PROCESSED_DIR = PROJECT_DIR / 'processed_dataset'
PLOTS_DIR = PROJECT_DIR / 'plots'

OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def parse_patient_summary(summary_path):
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    file_blocks = re.split(r'File Name:\s*', content)
    records = {}

    for block in file_blocks[1:]:
        lines = block.strip().split('\n')
        filename = lines[0].strip()
        seizure_intervals = []

        seizures_count = 0
        for line in lines:
            if 'Number of Seizures in File:' in line:
                match = re.search(r'\d+', line)
                if match:
                    seizures_count = int(match.group())

        starts = re.findall(r'Seizure (?:[0-9]+ )?Start Time:\s*([0-9]+)\s*seconds', block)
        ends = re.findall(r'Seizure (?:[0-9]+ )?End Time:\s*([0-9]+)\s*seconds', block)

        for s, e in zip(starts, ends):
            seizure_intervals.append((int(s), int(e)))

        records[filename] = {
            'num_seizures': seizures_count,
            'seizure_intervals': seizure_intervals
        }

    return records


def load_standard_18(edf_path):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    available = raw.ch_names
    selected = []

    for std_ch in STANDARD_CHANNELS:
        match = None
        for av in available:
            if av.upper().replace(' ', '') == std_ch.upper():
                match = av
                break
        if match is None:
            candidates = [av for av in available if av.upper().startswith(std_ch.upper() + '-')]
            if candidates:
                preferred = [c for c in candidates if c.endswith('-0')]
                match = preferred[0] if preferred else candidates[0]
        if match:
            selected.append(match)
        else:
            raise ValueError(f"Channel {std_ch} not found in {edf_path}")

    raw.pick(selected)
    raw.reorder_channels(selected)
    data = raw.get_data() * 1e6  # to microvolts
    sfreq = raw.info['sfreq']
    return data, sfreq


def preprocess_recording(data, sfreq=256.0):
    # 1. Chebyshev Type II Bandpass Filter (0.5 - 45 Hz)
    nyq = 0.5 * sfreq
    sos = signal.cheby2(N=4, rs=30, Wn=[0.5 / nyq, 45.0 / nyq], btype='bandpass', output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=-1)

    # 2. Decimate (256 Hz -> 128 Hz)
    decimated = signal.decimate(filtered, q=2, axis=-1, zero_phase=True)
    return decimated, sfreq / 2


def extract_epoch_features(epoch, sfreq=128.0):
    # 18 channels x 7 features = 126 features
    feat = []
    f, psd = signal.welch(epoch, fs=sfreq, nperseg=min(256, epoch.shape[-1]), axis=-1)
    for b_name, (l, h) in BANDS.items():
        idx_b = (f >= l) & (f <= h)
        bp = np.trapezoid(psd[:, idx_b], f[idx_b], axis=-1)  # shape (18,)
        feat.extend(bp)

    var = np.var(epoch, axis=-1)  # shape (18,)
    kurt = kurtosis(epoch, axis=-1)  # shape (18,)
    feat.extend(var)
    feat.extend(kurt)
    return feat


def run_full_cohort_pipeline(max_files_per_patient=3):
    print("=" * 75)
    print("CHB-MIT COHORT PREPROCESSING & MODEL TRAINING ENGINE (APPROACH 4)")
    print("Chebyshev II + Decimation + 126 Features + Random Forest Evaluation")
    print("=" * 75)

    # Find populated patient directories
    patient_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and len(list(d.glob('*.edf'))) > 0])
    print(f"\nDiscovered {len(patient_dirs)} populated patient folders.")

    all_features = []
    all_labels = []
    all_patient_ids = []
    all_recording_ids = []
    all_timestamps = []
    all_seizure_events = []

    file_audit_records = []
    total_start_time = time.time()

    for p_dir in patient_dirs:
        p_name = p_dir.name
        summary_path = p_dir / f'{p_name}-summary.txt'
        if not summary_path.exists():
            continue

        p_summary = parse_patient_summary(summary_path)

        # Separate seizure files and non-seizure files
        sz_files = [f for f, info in p_summary.items() if info['num_seizures'] > 0]
        normal_files = [f for f, info in p_summary.items() if info['num_seizures'] == 0]

        # Prioritize all seizure files up to max_files_per_patient, plus balanced normal file
        selected_files = sz_files[:max_files_per_patient]
        if normal_files and len(selected_files) < max_files_per_patient:
            selected_files.append(normal_files[0])

        if not selected_files:
            continue

        print(f"\nProcessing Subject: {p_name} ({len(selected_files)} recordings)")

        for fname in selected_files:
            edf_file = p_dir / fname
            if not edf_file.exists():
                continue

            sz_intervals = p_summary.get(fname, {}).get('seizure_intervals', [])

            t0 = time.time()
            try:
                raw_uV, sfreq = load_standard_18(edf_file)
            except Exception as e:
                print(f"  -> Skipping {fname}: {e}")
                continue

            dur_sec = raw_uV.shape[1] / sfreq
            decimated_uV, new_sfreq = preprocess_recording(raw_uV, sfreq=sfreq)
            t_proc = time.time() - t0

            # Windowing (4s windows, 2s step)
            win_samples = int(4.0 * new_sfreq)
            step_samples = int(2.0 * new_sfreq)
            n_samples = decimated_uV.shape[1]

            n_epochs = 0
            n_sz_epochs = 0

            # Track clinical seizure events in this file
            for sz_s, sz_e in sz_intervals:
                all_seizure_events.append({
                    'patient_id': p_name,
                    'recording_id': fname,
                    'start_sec': sz_s,
                    'end_sec': sz_e,
                    'duration_sec': sz_e - sz_s
                })

            for start_idx in range(0, n_samples - win_samples + 1, step_samples):
                end_idx = start_idx + win_samples
                epoch = decimated_uV[:, start_idx:end_idx]

                start_sec = start_idx / new_sfreq
                end_sec = end_idx / new_sfreq

                # Check seizure overlap
                is_seizure = 0
                for sz_s, sz_e in sz_intervals:
                    overlap_start = max(start_sec, sz_s)
                    overlap_end = min(end_sec, sz_e)
                    if overlap_end > overlap_start and (overlap_end - overlap_start) >= 2.0:
                        is_seizure = 1
                        break

                # Robust artifact check
                vars_ = np.var(epoch, axis=-1)
                diffs = np.diff(epoch, axis=-1)
                ptp = np.ptp(epoch, axis=-1)
                med_var = np.median(vars_)
                is_artifact = (
                    np.any(vars_ < 1.0) or 
                    np.max(np.abs(diffs)) > 200.0 or 
                    (med_var > 0 and np.max(vars_ / (med_var + 1e-6)) > 15.0 and np.max(vars_) > 4000) or
                    np.max(ptp) > 800.0
                )

                if is_artifact and is_seizure == 0:
                    continue  # screen artifact

                feat_vector = extract_epoch_features(epoch, sfreq=new_sfreq)

                all_features.append(feat_vector)
                all_labels.append(is_seizure)
                all_patient_ids.append(p_name)
                all_recording_ids.append(fname)
                all_timestamps.append((start_sec, end_sec))

                n_epochs += 1
                if is_seizure:
                    n_sz_epochs += 1

            file_audit_records.append({
                'patient_id': p_name,
                'recording_id': fname,
                'duration_hours': round(dur_sec / 3600, 2),
                'seizure_count': len(sz_intervals),
                'total_epochs': n_epochs,
                'seizure_epochs': n_sz_epochs,
                'proc_time_sec': round(t_proc, 2)
            })

            print(f"  [OK] {fname}: {n_epochs} epochs ({n_sz_epochs} ictal) | Processed in {t_proc:.2f}s")

    total_pipeline_time = time.time() - total_start_time

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    p_ids = np.array(all_patient_ids)
    rec_ids = np.array(all_recording_ids)

    print("\n" + "=" * 75)
    print("COHORT PREPROCESSING COMPLETED:")
    print("=" * 75)
    print(f"Total Epochs Extracted: {len(y)}")
    print(f"  - Interictal (Normal) Epochs: {np.sum(y == 0)}")
    print(f"  - Ictal (Seizure) Epochs: {np.sum(y == 1)}")
    print(f"  - Seizure Prevalence: {np.mean(y)*100:.2f}%")
    print(f"Feature Dimension: {X.shape[1]} features per window")
    print(f"Total Processing Time: {total_pipeline_time:.1f} seconds across cohort!")

    # Save processed arrays
    np.save(PROCESSED_DIR / 'X_cohort.npy', X)
    np.save(PROCESSED_DIR / 'y_cohort.npy', y)
    np.save(PROCESSED_DIR / 'patient_ids.npy', p_ids)
    np.save(PROCESSED_DIR / 'recording_ids.npy', rec_ids)

    df_manifest = pd.DataFrame(file_audit_records)
    df_manifest.to_csv(PROCESSED_DIR / 'metadata_cohort.csv', index=False)

    df_events = pd.DataFrame(all_seizure_events)
    df_events.to_csv(PROCESSED_DIR / 'seizure_events_metadata.csv', index=False)

    # -------------------------------------------------------------
    # Train and Evaluate Actual Machine Learning Classifier
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TRAINING & EVALUATING BENCHMARK CLASSIFIER (RANDOM FOREST):")
    print("=" * 75)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    logo = LeaveOneGroupOut()

    cv_probs = np.zeros(len(y))
    cv_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups=p_ids)):
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        cv_probs[val_idx] = probs
        # Default threshold 0.5 (unoptimized)
        cv_preds[val_idx] = (probs >= 0.5).astype(int)

    # 1. Epoch-level metrics
    epoch_sensitivity = recall_score(y, cv_preds) * 100
    epoch_precision = precision_score(y, cv_preds) * 100
    epoch_auc = roc_auc_score(y, cv_probs)
    epoch_auprc = average_precision_score(y, cv_probs)

    # 2. Event-level metrics & Detection Latency
    total_seizures_count = len(df_events)
    detected_seizures = 0
    latencies = []

    for _, row in df_events.iterrows():
        p_name = row['patient_id']
        r_name = row['recording_id']
        sz_start = row['start_sec']
        sz_end = row['end_sec']

        # Find indices of this seizure
        ev_indices = [
            i for i in range(len(y)) 
            if p_ids[i] == p_name and rec_ids[i] == r_name and all_timestamps[i][0] >= sz_start - 2.0 and all_timestamps[i][0] <= sz_end
        ]

        if not ev_indices:
            continue

        ev_preds = cv_preds[ev_indices]
        if np.any(ev_preds == 1):
            detected_seizures += 1
            first_hit = [all_timestamps[i][0] for i in ev_indices if cv_preds[i] == 1][0]
            lat = max(0.0, first_hit - sz_start + 2.0)
            latencies.append(lat)

    event_sensitivity = (detected_seizures / total_seizures_count) * 100 if total_seizures_count > 0 else 0
    mean_latency = np.mean(latencies) if latencies else 0.0

    # 3. False Alarm Rate (False Alarms per 24 Hours)
    total_hours_evaluated = df_manifest['duration_hours'].sum()
    false_positives = np.sum((cv_preds == 1) & (y == 0))
    far_per_hour = false_positives / total_hours_evaluated if total_hours_evaluated > 0 else 0
    far_per_day = far_per_hour * 24
    print("[NOTE] False-Alarm rate is calculated on a limited, seizure-rich subset (max 3 recordings per patient) and should be treated as an upper-bound estimate, not a deployment-ready metric.")

    metrics_summary = {
        'Cohort Subjects Evaluated': len(df_manifest['patient_id'].unique()),
        'Total Recordings Processed': len(df_manifest),
        'Total Evaluated Hours': round(total_hours_evaluated, 2),
        'Total Clinical Seizures': total_seizures_count,
        'Event-Level Sensitivity (%)': round(event_sensitivity, 2),
        'Epoch-Level Sensitivity (%)': round(epoch_sensitivity, 2),
        'Epoch-Level Precision (%)': round(epoch_precision, 2),
        'Mean Detection Latency (s)': round(mean_latency, 2),
        'False Alarms per 24 Hours': round(far_per_day, 2),
        'False Alarms per Hour': round(far_per_hour, 2),
        'AUROC': round(epoch_auc, 4),
        'AUPRC': round(epoch_auprc, 4),
        'Feature Representation': f"{X.shape[1]} features (7 features x 18 channels)"
    }

    df_report = pd.DataFrame([metrics_summary])
    df_report.to_csv(OUTPUT_DIR / 'cohort_model_evaluation_metrics.csv', index=False)

    print("\n" + "=" * 75)
    print("FINAL COHORT MODEL PERFORMANCE EVALUATION:")
    print("=" * 75)
    for k, v in metrics_summary.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------
    # Generate Cohort-Wide EDA Plots
    # -------------------------------------------------------------
    print("\nGenerating Cohort-Wide EDA Plots...")

    # Plot 1: Seizure Duration Distribution
    plt.figure(figsize=(8, 4.5), dpi=200)
    plt.hist(df_events['duration_sec'], bins=20, color='#ff7f0e', edgecolor='black', alpha=0.8)
    plt.axvline(df_events['duration_sec'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df_events["duration_sec"].median():.0f}s')
    plt.title(f'Clinical Seizure Duration Distribution Across Cohort (N={len(df_events)} Seizures)', fontsize=12, fontweight='bold')
    plt.xlabel('Seizure Duration (Seconds)', fontsize=11)
    plt.ylabel('Seizure Count', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cohort_seizure_durations.png')
    plt.close()

    # Plot 2: ROC and Precision-Recall Curves
    fpr, tpr, _ = roc_curve(y, cv_probs)
    prec, rec, _ = precision_recall_curve(y, cv_probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    axes[0].plot(fpr, tpr, color='#008080', linewidth=2.5, label=f'ROC Curve (AUROC = {epoch_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right')

    axes[1].plot(rec, prec, color='#d62728', linewidth=2.5, label=f'PR Curve (AUPRC = {epoch_auprc:.3f})')
    axes[1].axhline(np.mean(y), color='gray', linestyle='--', label=f'No-Skill Baseline ({np.mean(y)*100:.2f}%)')
    axes[1].set_title('Precision-Recall Curve (Under Severe Imbalance)', fontweight='bold')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')

    plt.suptitle('Trained Random Forest Classifier Evaluation (Cohort-Wide)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cohort_model_roc_pr.png')
    plt.close()

    print("[OK] All cohort EDA plots and metrics tables saved successfully!")
    return metrics_summary


if __name__ == '__main__':
    run_full_cohort_pipeline()
