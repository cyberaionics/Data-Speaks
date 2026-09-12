# Stage 06 — Classification

Paper-faithful SVM (RBF, γ=0.1, C=1) as the primary classifier.
Logistic Regression, Random Forest, and KNN are provided as
comparison baselines.
Evaluation: leave-one-record-out, per patient.

Run inference after preprocessing with:

	uv run python scripts/run_patient.py --patient chb01 --model logistic_regression

Compare the faster baseline models with:

	uv run python scripts/run_patient.py --patient chb01 --model all

Results are written to `results/benchmarks/<model>_<patient>_loso.json`.
Window-level predictions are written to `results/predictions/<model>/<patient>/`.
Each file contains per-record predictions summarized as sensitivity,
detection latency, and false detections per 24 hours. The SVM is included
in `--model all`, but may be substantially slower than the other models.

Rank completed model runs with:

	uv run python scripts/compare_models.py --patient chb01