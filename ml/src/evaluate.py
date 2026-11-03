"""
Model Evaluation utilities for Default and Tuned KNN models.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import TARGET_NAMES


def evaluate_model(model, X_test, y_test, model_name: str = "KNN Model") -> Dict[str, Any]:
    """
    Compute Accuracy, Confusion Matrix, and Classification Report.
    Labels: 0 = Rejected, 1 = Approved.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # cm format for labels [0, 1]:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Rejected", "Approved"],
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Rejected", "Approved"],
    )

    metrics = {
        "model_name": model_name,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),   # Correctly Rejected
        "false_positives": int(fp),  # False Approvals (High risk of default)
        "false_negatives": int(fn),  # False Rejections (Lost creditworthy customer)
        "true_positives": int(tp),   # Correctly Approved
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }

    return metrics, y_pred, y_proba

