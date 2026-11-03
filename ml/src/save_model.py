"""
Model serialization module.
Saves the best-performing complete preprocessing and KNN pipeline using Joblib,
along with a comprehensive JSON configuration file.
"""

import json
import os
import shutil
import joblib


def save_pipeline_and_config(
    model_pipeline,
    model_config: dict,
    models_dir: str = "ml/models",
    backend_dir: str = "backend",
):
    """
    Serializes pipeline and JSON config to models directory and backend directory.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(backend_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "best_knn_model.joblib")
    config_path = os.path.join(models_dir, "model_config.json")

    # Serialize trained pipeline with joblib
    joblib.dump(model_pipeline, model_path)
    print(f"[OK] Saved model pipeline to: {model_path}")

    # Write JSON metadata
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)
    print(f"[OK] Saved model configuration to: {config_path}")

    # Copy directly into backend directory for production serving
    backend_model_path = os.path.join(backend_dir, "best_knn_model.joblib")
    backend_config_path = os.path.join(backend_dir, "model_config.json")
    shutil.copyfile(model_path, backend_model_path)
    shutil.copyfile(config_path, backend_config_path)
    print(f"[OK] Synced model and config to backend directory: {backend_dir}")

    return model_path, config_path

