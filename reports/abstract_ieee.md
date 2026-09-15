# Abstract

This project develops a patient-specific seizure detection pipeline for the CHB-MIT Scalp EEG Database, focusing on Patient 1 (CHB01). The system begins with raw EDF signals, performs channel selection, signal filtering, artifact removal through ICA, per-channel normalization, and segmentation into fixed 6-second windows. Spectral features are extracted from the resulting windows and used to evaluate seizure detection using leave-one-record-out cross-validation.

Several classifiers were compared, including Logistic Regression, Random Forest, KNN, and SVM, with Random Forest providing the strongest balance between seizure sensitivity and false detections. The study also includes clustering and dimensionality reduction analyses to interpret the structure of the spectral feature space and identify the frequency patterns most associated with ictal windows.

The primary goal is to build an interpretable and auditable EEG seizure detection workflow that can be reproduced and compared across teams using the same patient data, preprocessing rules, and evaluation protocol. The project demonstrates how preprocessing choices, labeling logic, and event-level metrics influence the final detection outcome and provides reproducible artifacts such as processed arrays, benchmark summaries, figures, and CSV exports for reporting and submission.

The dataset comprises 42 EDF recordings from Patient CHB01, with 7 containing seizure annotations and 35 non-seizure recordings. After preprocessing and windowing, the dataset contains 36,343 total windows (112 ictal, 36,231 interictal) with 17 retained channels and 408 features per window. Three classifiers were evaluated: Logistic Regression, Random Forest, KNN, and SVM.

Random Forest was selected as the model providing the strongest balance between seizure sensitivity and false detections, achieving 100% sensitivity with a mean latency of 5.86 seconds and 2.06 false detections per 24 hours, while Logistic Regression achieved 100% sensitivity but with 18.07 false detections per 24 hours.

The project demonstrates the importance of proper event-level metrics (rather than window-level accuracy) for imbalanced seizure detection data, and provides a framework for team comparison using standardized preprocessing, LOSO validation, and event-level metric reporting.
