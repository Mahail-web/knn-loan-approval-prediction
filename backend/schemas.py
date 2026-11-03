"""
Pydantic schemas and request/response models for Loan Prediction API.
"""

from typing import Dict, Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator


class LoanApplicationRequest(BaseModel):
    Gender: Literal["Male", "Female"] = Field(
        ..., description="Applicant gender ('Male' or 'Female')"
    )
    Married: Literal["Yes", "No"] = Field(
        ..., description="Applicant marital status ('Yes' or 'No')"
    )
    Dependents: Literal["0", "1", "2", "3+"] = Field(
        ..., description="Number of dependents ('0', '1', '2', or '3+')"
    )
    Education: Literal["Graduate", "Not Graduate"] = Field(
        ..., description="Applicant education level ('Graduate' or 'Not Graduate')"
    )
    Self_Employed: Literal["Yes", "No"] = Field(
        ..., description="Whether applicant is self-employed ('Yes' or 'No')"
    )
    ApplicantIncome: float = Field(
        ..., ge=0, description="Monthly applicant income in USD/currency units"
    )
    CoapplicantIncome: float = Field(
        0.0, ge=0, description="Monthly co-applicant income in USD/currency units"
    )
    LoanAmount: float = Field(
        ..., gt=0, description="Requested loan amount (in thousands, e.g., 128 for $128,000)"
    )
    Loan_Amount_Term: float = Field(
        360.0, gt=0, description="Term of loan in months (e.g., 360 for 30 years, 180 for 15 years)"
    )
    Credit_History: float = Field(
        ..., description="Credit history guidelines met (1.0 = meets guidelines, 0.0 = does not meet)"
    )
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = Field(
        ..., description="Location type of the property ('Urban', 'Semiurban', 'Rural')"
    )

    @field_validator("Credit_History")
    @classmethod
    def validate_credit_history(cls, v: float) -> float:
        if v not in (0.0, 1.0, 0, 1):
            raise ValueError("Credit_History must be either 1.0 (meets criteria) or 0.0 (does not meet criteria)")
        return float(v)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "1",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 4583,
                "CoapplicantIncome": 1508.0,
                "LoanAmount": 128.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "Property_Area": "Semiurban",
            }
        }
    }


class PredictionResponse(BaseModel):
    success: bool
    prediction: Literal["Approved", "Rejected"]
    prediction_code: Literal["Y", "N"]
    message: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: Dict[str, float]
    applicant_summary: Dict[str, Any]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: Optional[str] = None
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    selected_k: int
    best_hyperparameters: Dict[str, Any]
    test_accuracy: float
    best_cv_accuracy: float
    training_features: list[str]
    categorical_features: list[str]
    numerical_features: list[str]
    metrics_breakdown: Dict[str, Any]
    dataset_metadata: Dict[str, Any]

