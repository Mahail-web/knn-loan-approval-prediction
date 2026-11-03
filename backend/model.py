"""
Model loader and inference wrapper for the trained K-Nearest Neighbors pipeline.
"""

import json
import os
from typing import Dict, Any, Tuple
import joblib
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "best_knn_model.joblib")
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "model_config.json")


class LoanModelManager:
    """
    Singleton manager for loading and serving the trained KNN model pipeline.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, config_path: str = DEFAULT_CONFIG_PATH):
        self.model_path = model_path
        self.config_path = config_path
        self.pipeline = None
        self.config: Dict[str, Any] = {}
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        """Loads the serialized pipeline and config JSON from disk."""
        if not os.path.exists(self.model_path):
            # Fallback to ml/models if not directly in backend
            alt_path = os.path.join(CURRENT_DIR, "..", "ml", "models", "best_knn_model.joblib")
            if os.path.exists(alt_path):
                self.model_path = alt_path

        if not os.path.exists(self.config_path):
            alt_cfg = os.path.join(CURRENT_DIR, "..", "ml", "models", "model_config.json")
            if os.path.exists(alt_cfg):
                self.config_path = alt_cfg

        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            self.is_loaded = True
            print(f"[OK] Model pipeline loaded from: {self.model_path}")
        else:
            print(f"[WARN] Model pipeline file not found at: {self.model_path}")

        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            print(f"[OK] Model config loaded from: {self.config_path}")
        else:
            print(f"[WARN] Model config file not found at: {self.config_path}")

    def predict(self, input_dict: Dict[str, Any]) -> Tuple[str, str, float, Dict[str, float]]:
        """
        Runs applicant information through the saved Scikit-learn Pipeline.
        Returns:
            (prediction_label, prediction_code, confidence, class_probabilities)
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        # Ensure input is organized in expected DataFrame format
        feature_order = self.config.get("training_feature_names", [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "Property_Area", "ApplicantIncome", "CoapplicantIncome",
            "LoanAmount", "Loan_Amount_Term", "Credit_History"
        ])
        
        # Build single-row DataFrame
        df_input = pd.DataFrame([input_dict])
        # Reorder/filter columns to match training features exactly
        df_input = df_input[feature_order]

        # Pipeline handles imputer, onehot, and scaler automatically!
        raw_pred = self.pipeline.predict(df_input)[0]
        
        # Binary target: 1 = Approved ('Y'), 0 = Rejected ('N')
        pred_code = "Y" if raw_pred == 1 else "N"
        pred_label = "Approved" if raw_pred == 1 else "Rejected"

        # Calculate class probabilities
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_input)[0]
            # classes_ typically [0, 1]
            classes = self.pipeline.classes_
            prob_dict = {}
            for cls_idx, cls_val in enumerate(classes):
                label = "Approved" if cls_val == 1 else "Rejected"
                prob_dict[label] = round(float(probs[cls_idx]), 4)
            confidence = prob_dict.get(pred_label, 0.5)
        else:
            confidence = 1.0
            prob_dict = {"Approved": 1.0 if raw_pred == 1 else 0.0, "Rejected": 0.0 if raw_pred == 1 else 1.0}

        return pred_label, pred_code, confidence, prob_dict


# Global model manager instance
model_manager = LoanModelManager()

