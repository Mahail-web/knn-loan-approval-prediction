"""
Master Orchestration Script for Loan Approval Prediction KNN Capstone.
Executes Phases 1 through 7 seamlessly.
"""

import json
import os
import sys
import pandas as pd

# Ensure src directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_model
from preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_MAPPING,
    TARGET_NAMES,
    load_data,
    prepare_features_and_target,
    split_data,
)
from save_model import save_pipeline_and_config
from train import (
    evaluate_specific_k_values,
    train_default_knn,
    tune_knn_hyperparameters,
)
from visualize import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_k_value_comparison,
    plot_loan_approval_distribution,
    plot_missing_values,
    plot_numerical_distributions,
)


def run_full_pipeline(
    data_path: str = "ml/data/train.csv",
    vis_dir: str = "ml/visualizations",
    models_dir: str = "ml/models",
    backend_dir: str = "backend",
):
    print("=" * 70)
    print("LOAN APPROVAL PREDICTION USING KNN - END-TO-END PIPELINE")
    print("=" * 70)

    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(backend_dir, exist_ok=True)

    # -------------------------------------------------------------
    # PHASE 1: DATASET ACQUISITION AND EXPLORATION
    # -------------------------------------------------------------
    print("\n>>> PHASE 1: DATASET ACQUISITION & EXPLORATION")
    df = load_data(data_path)
    print(f"Dataset Dimensions (shape): {df.shape}")
    print("\nDataset Column Names & Data Types:")
    print(df.dtypes)
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    print(f"\nDuplicate Records Count: {df.duplicated().sum()}")
    print("\nTarget Variable 'Loan_Status' Distribution:")
    print(df["Loan_Status"].value_counts())
    print(df["Loan_Status"].value_counts(normalize=True) * 100)

    # -------------------------------------------------------------
    # PHASE 2: DATA PREPROCESSING
    # -------------------------------------------------------------
    print("\n>>> PHASE 2: DATA PREPROCESSING & FEATURE SEPARATION")
    X, y = prepare_features_and_target(df)
    print(f"Features matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(f"Target classes: {TARGET_NAMES}")

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set:  X_test={X_test.shape}, y_test={y_test.shape}")
    print("Target distribution in test set:")
    print(y_test.value_counts())

    # -------------------------------------------------------------
    # PHASE 3: TRAIN DEFAULT KNN & K=3, 5, 7 COMPARISON
    # -------------------------------------------------------------
    print("\n>>> PHASE 3: DEFAULT KNN & K-VALUE EXPERIMENT")
    default_model = train_default_knn(X_train, y_train)
    default_metrics, default_preds, default_proba = evaluate_model(
        default_model, X_test, y_test, model_name="Default KNN (K=5)"
    )
    print(f"Default KNN (K=5) Test Accuracy: {default_metrics['accuracy']:.4f}")
    print("\nDefault KNN Classification Report:")
    print(default_metrics["classification_report_text"])

    # Comparison for K=3, 5, 7
    k_comparison, k_models = evaluate_specific_k_values(
        X_train, y_train, X_test, y_test, k_values=[3, 5, 7]
    )
    print("\nK-Value Comparison Table:")
    print("| K Value | Accuracy |")
    print("|---------|----------|")
    for k_val, acc in k_comparison.items():
        print(f"| {k_val:<7} | {acc * 100:.2f}%   |")

    # -------------------------------------------------------------
    # PHASE 4: HYPERPARAMETER TUNING VIA GRIDSEARCHCV
    # -------------------------------------------------------------
    print("\n>>> PHASE 4: HYPERPARAMETER TUNING (K = 1 to 20)")
    grid_search = tune_knn_hyperparameters(X_train, y_train, k_range=range(1, 21), cv_folds=5)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best 5-Fold Cross-Validation Accuracy: {best_cv_score * 100:.2f}%")

    # Extract CV curve for n_neighbors
    cv_df = pd.DataFrame(grid_search.cv_results_)
    # Filter for best metric/weights to get clean curve across n_neighbors
    best_metric = best_params.get("classifier__metric", "euclidean")
    best_weights = best_params.get("classifier__weights", "uniform")
    subset_cv = cv_df[
        (cv_df["param_classifier__metric"] == best_metric)
        & (cv_df["param_classifier__weights"] == best_weights)
    ].sort_values("param_classifier__n_neighbors")

    k_list = subset_cv["param_classifier__n_neighbors"].tolist()
    mean_cv_scores = subset_cv["mean_test_score"].tolist()
    cv_results_dict = {
        "k_list": [int(k) for k in k_list],
        "mean_cv_scores": [float(s) for s in mean_cv_scores],
        "best_k": int(best_params["classifier__n_neighbors"]),
        "best_score": float(best_cv_score),
    }

    # -------------------------------------------------------------
    # PHASE 5: MODEL EVALUATION (TUNED VS DEFAULT)
    # -------------------------------------------------------------
    print("\n>>> PHASE 5: EVALUATION ON UNTOUCHED TEST SET")
    best_k_val = best_params["classifier__n_neighbors"]
    tuned_metrics, tuned_preds, tuned_proba = evaluate_model(
        best_estimator, X_test, y_test, model_name=f"Tuned KNN (K={best_k_val})"
    )

    print(f"\n--- Comparison Summary ---")
    print(f"Default Model Accuracy: {default_metrics['accuracy'] * 100:.2f}%")
    print(f"Tuned Model Accuracy:   {tuned_metrics['accuracy'] * 100:.2f}%")
    print("\nTuned Model Classification Report:")
    print(tuned_metrics["classification_report_text"])

    print("Risk Analysis Breakdown on Test Set (123 total test applicants):")
    print(f"- Correctly Approved (True Positives):   {tuned_metrics['true_positives']}")
    print(f"- Correctly Rejected (True Negatives):   {tuned_metrics['true_negatives']}")
    print(f"- False Approvals (False Positives - High Default Risk): {tuned_metrics['false_positives']}")
    print(f"- False Rejections (False Negatives - Lost Opportunity): {tuned_metrics['false_negatives']}")

    # -------------------------------------------------------------
    # PHASE 6: DATA VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n>>> PHASE 6: GENERATING 7 PROFESSIONAL VISUALIZATIONS")
    vis_paths = []
    vis_paths.append(plot_loan_approval_distribution(df, vis_dir))
    vis_paths.append(plot_missing_values(df, vis_dir))
    vis_paths.append(plot_numerical_distributions(df, vis_dir))
    vis_paths.append(plot_correlation_heatmap(df, vis_dir))
    vis_paths.append(plot_k_value_comparison(k_comparison, cv_results_dict, vis_dir))
    vis_paths.append(
        plot_confusion_matrix(
            default_metrics["confusion_matrix"],
            "Default KNN (K=5)",
            "confusion_matrix_default_knn.png",
            vis_dir,
        )
    )
    vis_paths.append(
        plot_confusion_matrix(
            tuned_metrics["confusion_matrix"],
            f"Optimized KNN (K={best_k_val})",
            "confusion_matrix_optimized_knn.png",
            vis_dir,
        )
    )
    for p in vis_paths:
        print(f"[OK] Generated: {p}")

    # -------------------------------------------------------------
    # PHASE 7: SERIALIZE MODEL AND CONFIG
    # -------------------------------------------------------------
    print("\n>>> PHASE 7: SERIALIZING BEST MODEL & METADATA")
    model_config = {
        "model_name": "K-Nearest Neighbors Classifier",
        "model_version": "1.0.0",
        "algorithm": "KNeighborsClassifier",
        "selected_k": int(best_k_val),
        "best_hyperparameters": {
            "n_neighbors": int(best_k_val),
            "weights": str(best_params["classifier__weights"]),
            "metric": str(best_params["classifier__metric"]),
        },
        "test_accuracy": float(tuned_metrics["accuracy"]),
        "default_test_accuracy": float(default_metrics["accuracy"]),
        "best_cv_accuracy": float(best_cv_score),
        "k_value_experiments": {
            str(k): float(acc) for k, acc in k_comparison.items()
        },
        "training_feature_names": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target_mapping": TARGET_MAPPING,
        "target_names": TARGET_NAMES,
        "confusion_matrix": tuned_metrics["confusion_matrix"],
        "metrics_breakdown": {
            "true_positives": tuned_metrics["true_positives"],
            "true_negatives": tuned_metrics["true_negatives"],
            "false_positives": tuned_metrics["false_positives"],
            "false_negatives": tuned_metrics["false_negatives"],
            "classification_report": tuned_metrics["classification_report_dict"],
        },
        "dataset_metadata": {
            "total_samples": int(df.shape[0]),
            "total_features": int(df.shape[1]),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "approval_rate": float(df["Loan_Status"].value_counts(normalize=True)["Y"]),
        },
    }

    save_pipeline_and_config(
        model_pipeline=best_estimator,
        model_config=model_config,
        models_dir=models_dir,
        backend_dir=backend_dir,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return model_config


if __name__ == "__main__":
    run_full_pipeline()
