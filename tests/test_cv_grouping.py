import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut

def test_leave_one_group_out_splits():
    # Load processed arrays from the repo's processed_dataset/ folder
    processed_dir = Path(__file__).resolve().parent.parent / 'processed_dataset'
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"'{processed_dir}' not found. Run `python scripts/cohort_pipeline.py` first "
            "to generate the processed dataset before running this test."
        )
    X = np.load(processed_dir / 'X_cohort.npy')
    y = np.load(processed_dir / 'y_cohort.npy')
    p_ids = np.load(processed_dir / 'patient_ids.npy')
    rec_ids = np.load(processed_dir / 'recording_ids.npy')

    logo = LeaveOneGroupOut()
    for _, (train_idx, val_idx) in enumerate(logo.split(X, y, groups=p_ids)):
        # Ensure no patient appears in both train and validation
        train_patients = set(p_ids[train_idx])
        val_patients = set(p_ids[val_idx])
        assert train_patients.isdisjoint(val_patients), "Patient ID leakage between train and validation folds"
        # Ensure all epochs from a recording stay in the same fold
        val_rec_set = set(rec_ids[val_idx])
        for rec in val_rec_set:
            rec_mask = rec_ids == rec
            idx_in_val = np.where(rec_mask)[0]
            assert np.all(np.isin(idx_in_val, val_idx)), "Recording ID split across folds"

if __name__ == "__main__":
    test_leave_one_group_out_splits()
    print("All CV grouping tests passed.")
