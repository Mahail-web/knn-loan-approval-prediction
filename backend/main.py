"""
FastAPI application for Loan Approval Prediction using KNN.
Exposes RESTful endpoints for model health, metadata, and applicant prediction.
"""

from datetime import datetime, timezone
import os
import sys

# Ensure backend directory is in sys.path regardless of execution working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import model_manager
from schemas import (
    HealthResponse,
    LoanApplicationRequest,
    ModelInfoResponse,
    PredictionResponse,
)

app = FastAPI(
    title="LoanWise AI - Loan Approval Prediction API",
    description="Production-ready REST API serving a trained K-Nearest Neighbors (KNN) classifier for automated loan risk evaluation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
allowed_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,*")
origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def read_root():
    """Welcome endpoint with API documentation and status links."""
    return {
        "project": "LoanWise AI – Loan Approval Prediction System",
        "description": "FastAPI service serving an optimized K-Nearest Neighbors (KNN) classifier.",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Returns the backend health and model loading status."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        model_name=model_manager.config.get("model_name", "K-Nearest Neighbors Classifier"),
        version=model_manager.config.get("model_version", "1.0.0"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """Returns model metadata, chosen K value, hyperparameters, and evaluation metrics."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently not loaded on the server.",
        )

    cfg = model_manager.config
    return ModelInfoResponse(
        model_name=cfg.get("model_name", "K-Nearest Neighbors Classifier"),
        model_version=cfg.get("model_version", "1.0.0"),
        algorithm=cfg.get("algorithm", "KNeighborsClassifier"),
        selected_k=cfg.get("selected_k", 19),
        best_hyperparameters=cfg.get("best_hyperparameters", {}),
        test_accuracy=cfg.get("test_accuracy", 0.8537),
        best_cv_accuracy=cfg.get("best_cv_accuracy", 0.7983),
        training_features=cfg.get("training_feature_names", []),
        categorical_features=cfg.get("categorical_features", []),
        numerical_features=cfg.get("numerical_features", []),
        metrics_breakdown=cfg.get("metrics_breakdown", {}),
        dataset_metadata=cfg.get("dataset_metadata", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_loan_approval(application: LoanApplicationRequest):
    """
    Accepts applicant information, processes features through the serialized
    Scikit-learn pipeline, and returns loan approval classification and probabilities.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not initialized. Please ensure the model file is available.",
        )

    try:
        input_data = application.model_dump()
        pred_label, pred_code, confidence, class_probs = model_manager.predict(input_data)

        if pred_code == "Y":
            message = "The model predicts that this loan application will be approved based on neighbor similarity patterns."
        else:
            message = "The model predicts that this loan application will be rejected based on applicant risk profile and neighbor patterns."

        return PredictionResponse(
            success=True,
            prediction=pred_label,
            prediction_code=pred_code,
            message=message,
            confidence=confidence,
            class_probabilities=class_probs,
            applicant_summary={
                "applicant_income": input_data["ApplicantIncome"],
                "coapplicant_income": input_data["CoapplicantIncome"],
                "total_income": input_data["ApplicantIncome"] + input_data["CoapplicantIncome"],
                "loan_amount_k": input_data["LoanAmount"],
                "term_months": input_data["Loan_Amount_Term"],
                "credit_history": "Meets Guidelines (1.0)" if input_data["Credit_History"] == 1.0 else "Does Not Meet (0.0)",
                "property_area": input_data["Property_Area"],
                "education": input_data["Education"],
            },
            disclaimer="This application demonstrates machine learning-based loan approval prediction using K-Nearest Neighbors. Its output is an algorithmic estimation and does not constitute a legal, formal, or binding financial commitment.",
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid applicant data: {str(ve)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the loan approval prediction. Please check inputs and retry.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
