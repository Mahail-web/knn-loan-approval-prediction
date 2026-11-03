import { LoanApplicationData } from "@/types/loan";

export interface ValidationErrors {
  [key: string]: string;
}

export function validateLoanForm(data: LoanApplicationData): ValidationErrors {
  const errors: ValidationErrors = {};

  if (!data.Gender) {
    errors.Gender = "Please select applicant gender.";
  }

  if (!data.Married) {
    errors.Married = "Please select marital status.";
  }

  if (data.Dependents === undefined || data.Dependents === null) {
    errors.Dependents = "Please specify number of dependents.";
  }

  if (!data.Education) {
    errors.Education = "Please select education level.";
  }

  if (!data.Self_Employed) {
    errors.Self_Employed = "Please indicate if self-employed.";
  }

  if (isNaN(data.ApplicantIncome) || data.ApplicantIncome === null || data.ApplicantIncome === undefined) {
    errors.ApplicantIncome = "Applicant income is required.";
  } else if (data.ApplicantIncome < 0) {
    errors.ApplicantIncome = "Applicant income cannot be negative.";
  }

  if (isNaN(data.CoapplicantIncome) || data.CoapplicantIncome === null || data.CoapplicantIncome === undefined) {
    errors.CoapplicantIncome = "Co-applicant income must be 0 or greater.";
  } else if (data.CoapplicantIncome < 0) {
    errors.CoapplicantIncome = "Co-applicant income cannot be negative.";
  }

  if (isNaN(data.LoanAmount) || data.LoanAmount === null || data.LoanAmount === undefined || data.LoanAmount <= 0) {
    errors.LoanAmount = "Please enter a valid loan amount greater than 0 (in thousands).";
  }

  if (
    isNaN(data.Loan_Amount_Term) ||
    data.Loan_Amount_Term === null ||
    data.Loan_Amount_Term === undefined ||
    data.Loan_Amount_Term <= 0
  ) {
    errors.Loan_Amount_Term = "Please enter loan term in months (e.g., 360 for 30 years).";
  }

  if (data.Credit_History !== 1.0 && data.Credit_History !== 0.0) {
    errors.Credit_History = "Please select credit history eligibility.";
  }

  if (!data.Property_Area) {
    errors.Property_Area = "Please select property area type.";
  }

  return errors;
}

