# Abstract

This project develops a patient-specific seizure detection pipeline for the CHB-MIT Scalp EEG Database, focusing on Patient 1 (CHB01). The system begins with raw EDF signals, performs channel selection, signal filtering, artifact removal through ICA, per-channel normalization, and segmentation into fixed 6-second windows. Spectral features are extracted from the resulting windows and used to evaluate seizure detection using leave-one-record-out cross-validation.

Several classifiers were compared, including Logistic Regression, Random Forest, KNN, and SVM, with Random Forest providing the strongest balance between seizure sensitivity and false detections. The study also includes clustering and dimensionality reduction analyses to interpret the structure of the spectral feature space and identify the frequency patterns most associated with ictal windows.

The primary goal is to build an interpretable and auditable EEG seizure detection workflow that can be reproduced and compared across teams using the same patient data, preprocessing rules, and evaluation protocol. The project demonstrates how preprocessing choices, labeling logic, and event-level metrics influence the final detection outcome and provides reproducible artifacts such as processed arrays, benchmark summaries, figures, and CSV exports for reporting and midterm submission.
