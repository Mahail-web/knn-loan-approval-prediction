export type GenderType = "Male" | "Female";
export type MarriedType = "Yes" | "No";
export type DependentsType = "0" | "1" | "2" | "3+";
export type EducationType = "Graduate" | "Not Graduate";
export type SelfEmployedType = "Yes" | "No";
export type PropertyAreaType = "Urban" | "Semiurban" | "Rural";
export type CreditHistoryType = 1.0 | 0.0;

export interface LoanApplicationData {
  Gender: GenderType;
  Married: MarriedType;
  Dependents: DependentsType;
  Education: EducationType;
  Self_Employed: SelfEmployedType;
  ApplicantIncome: number;
  CoapplicantIncome: number;
  LoanAmount: number;
  Loan_Amount_Term: number;
  Credit_History: CreditHistoryType;
  Property_Area: PropertyAreaType;
}

export interface PredictionResponse {
  success: boolean;
  prediction: "Approved" | "Rejected";
  prediction_code: "Y" | "N";
  message: string;
  confidence: number;
  class_probabilities: {
    Approved: number;
    Rejected: number;
  };
  applicant_summary: {
    applicant_income: number;
    coapplicant_income: number;
    total_income: number;
    loan_amount_k: number;
    term_months: number;
    credit_history: string;
    property_area: string;
    education: string;
  };
  disclaimer: string;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  algorithm: string;
  selected_k: number;
  best_hyperparameters: {
    n_neighbors: number;
    weights: string;
    metric: string;
  };
  test_accuracy: number;
  best_cv_accuracy: number;
  training_features: string[];
  categorical_features: string[];
  numerical_features: string[];
  metrics_breakdown: {
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    classification_report: {
      Rejected: { precision: number; recall: number; "f1-score": number; support: number };
      Approved: { precision: number; recall: number; "f1-score": number; support: number };
      accuracy: number;
    };
  };
  dataset_metadata: {
    total_samples: number;
    total_features: number;
    train_samples: number;
    test_samples: number;
    approval_rate: number;
  };
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_name?: string;
  version?: string;
  timestamp: string;
}

