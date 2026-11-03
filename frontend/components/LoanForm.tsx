"use client";

import React, { useState } from "react";
import {
  Send,
  RotateCcw,
  Sparkles,
  AlertCircle,
  HelpCircle,
  DollarSign,
  User,
  Home,
  CheckCircle2,
} from "lucide-react";
import { LoanApplicationData, PredictionResponse } from "@/types/loan";
import { submitLoanPrediction } from "@/lib/api";
import { validateLoanForm, ValidationErrors } from "@/lib/validation";

interface LoanFormProps {
  onPredictionReceived: (result: PredictionResponse | null) => void;
  isLoading: boolean;
  setIsLoading: (val: boolean) => void;
  apiError: string | null;
  setApiError: (val: string | null) => void;
}

const INITIAL_FORM_DATA: LoanApplicationData = {
  Gender: "Male",
  Married: "Yes",
  Dependents: "0",
  Education: "Graduate",
  Self_Employed: "No",
  ApplicantIncome: 5000,
  CoapplicantIncome: 1500,
  LoanAmount: 120,
  Loan_Amount_Term: 360,
  Credit_History: 1.0,
  Property_Area: "Semiurban",
};

export default function LoanForm({
  onPredictionReceived,
  isLoading,
  setIsLoading,
  apiError,
  setApiError,
}: LoanFormProps) {
  const [formData, setFormData] = useState<LoanApplicationData>(INITIAL_FORM_DATA);
  const [errors, setErrors] = useState<ValidationErrors>({});

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    let parsedValue: any = value;

    if (type === "number") {
      parsedValue = value === "" ? "" : parseFloat(value);
    } else if (name === "Credit_History") {
      parsedValue = parseFloat(value);
    }

    setFormData((prev) => ({
      ...prev,
      [name]: parsedValue,
    }));

    // Clear specific field error
    if (errors[name]) {
      setErrors((prev) => {
        const updated = { ...prev };
        delete updated[name];
        return updated;
      });
    }
  };

  const loadPreset = (preset: "approved" | "rejected") => {
    setApiError(null);
    setErrors({});
    if (preset === "approved") {
      setFormData({
        Gender: "Male",
        Married: "Yes",
        Dependents: "1",
        Education: "Graduate",
        Self_Employed: "No",
        ApplicantIncome: 6500,
        CoapplicantIncome: 2200,
        LoanAmount: 135,
        Loan_Amount_Term: 360,
        Credit_History: 1.0,
        Property_Area: "Semiurban",
      });
    } else {
      setFormData({
        Gender: "Female",
        Married: "No",
        Dependents: "0",
        Education: "Not Graduate",
        Self_Employed: "Yes",
        ApplicantIncome: 1400,
        CoapplicantIncome: 0,
        LoanAmount: 380,
        Loan_Amount_Term: 180,
        Credit_History: 0.0,
        Property_Area: "Rural",
      });
    }
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM_DATA);
    setErrors({});
    setApiError(null);
    onPredictionReceived(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);

    const validationErrors = validateLoanForm(formData);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    setIsLoading(true);
    try {
      const response = await submitLoanPrediction(formData);
      onPredictionReceived(response);

      // Smooth scroll to results
      setTimeout(() => {
        const resultSection = document.getElementById("prediction-result");
        if (resultSection) {
          resultSection.scrollIntoView({ behavior: "smooth" });
        }
      }, 100);
    } catch (err: any) {
      setApiError(err.message || "Failed to communicate with prediction backend.");
      onPredictionReceived(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="loan-form" className="bg-white rounded-2xl border border-slate-200 shadow-xl shadow-slate-100/50 p-6 sm:p-8 md:p-10">
      {/* Header & Preset controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-6 mb-8 border-b border-slate-100 gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
            <span>Applicant Credit Application Form</span>
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            Fill in applicant details to evaluate loan eligibility using our trained KNN model.
          </p>
        </div>

        {/* Quick Test Presets */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider hidden lg:inline">
            Load Test Case:
          </span>
          <button
            type="button"
            onClick={() => loadPreset("approved")}
            className="px-3 py-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors flex items-center gap-1.5"
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Prime Applicant (Y)</span>
          </button>
          <button
            type="button"
            onClick={() => loadPreset("rejected")}
            className="px-3 py-1.5 text-xs font-medium text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors flex items-center gap-1.5"
          >
            <AlertCircle className="w-3.5 h-3.5" />
            <span>High Risk Applicant (N)</span>
          </button>
        </div>
      </div>

      {/* Global API Error Alert */}
      {apiError && (
        <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            <p className="font-semibold">Backend Error</p>
            <p className="mt-0.5">{apiError}</p>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* SECTION 1: Personal & Demographic Info */}
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
            <User className="w-4 h-4 text-blue-600" />
            <span>1. Demographics & Background</span>
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {/* Gender */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Gender <span className="text-rose-500">*</span>
              </label>
              <select
                name="Gender"
                value={formData.Gender}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
              {errors.Gender && <p className="text-xs text-rose-500 mt-1">{errors.Gender}</p>}
            </div>

            {/* Married */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Marital Status <span className="text-rose-500">*</span>
              </label>
              <select
                name="Married"
                value={formData.Married}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              >
                <option value="Yes">Yes (Married)</option>
                <option value="No">No (Single/Unmarried)</option>
              </select>
              {errors.Married && <p className="text-xs text-rose-500 mt-1">{errors.Married}</p>}
            </div>

            {/* Dependents */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Number of Dependents <span className="text-rose-500">*</span>
              </label>
              <select
                name="Dependents"
                value={formData.Dependents}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              >
                <option value="0">0 (None)</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3+">3+ (Three or more)</option>
              </select>
              {errors.Dependents && <p className="text-xs text-rose-500 mt-1">{errors.Dependents}</p>}
            </div>

            {/* Education */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Education Level <span className="text-rose-500">*</span>
              </label>
              <select
                name="Education"
                value={formData.Education}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              >
                <option value="Graduate">Graduate</option>
                <option value="Not Graduate">Not Graduate</option>
              </select>
              {errors.Education && <p className="text-xs text-rose-500 mt-1">{errors.Education}</p>}
            </div>

            {/* Self Employed */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Self Employed <span className="text-rose-500">*</span>
              </label>
              <select
                name="Self_Employed"
                value={formData.Self_Employed}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              >
                <option value="No">No (Salaried)</option>
                <option value="Yes">Yes (Self-Employed)</option>
              </select>
              {errors.Self_Employed && <p className="text-xs text-rose-500 mt-1">{errors.Self_Employed}</p>}
            </div>

            {/* Property Area */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Property Area <span className="text-rose-500">*</span>
              </label>
              <select
                name="Property_Area"
                value={formData.Property_Area}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              >
                <option value="Semiurban">Semiurban</option>
                <option value="Urban">Urban</option>
                <option value="Rural">Rural</option>
              </select>
              {errors.Property_Area && <p className="text-xs text-rose-500 mt-1">{errors.Property_Area}</p>}
            </div>
          </div>
        </div>

        {/* SECTION 2: Financial & Credit Information */}
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
            <DollarSign className="w-4 h-4 text-emerald-600" />
            <span>2. Financials & Credit Metrics</span>
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {/* Applicant Income */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Applicant Monthly Income ($) <span className="text-rose-500">*</span>
              </label>
              <input
                type="number"
                name="ApplicantIncome"
                value={formData.ApplicantIncome}
                onChange={handleChange}
                min="0"
                step="50"
                placeholder="e.g. 5000"
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              />
              <p className="text-[11px] text-slate-400 mt-1">Primary monthly earnings</p>
              {errors.ApplicantIncome && (
                <p className="text-xs text-rose-500 mt-1">{errors.ApplicantIncome}</p>
              )}
            </div>

            {/* Coapplicant Income */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Co-Applicant Monthly Income ($)
              </label>
              <input
                type="number"
                name="CoapplicantIncome"
                value={formData.CoapplicantIncome}
                onChange={handleChange}
                min="0"
                step="50"
                placeholder="e.g. 1500 (or 0)"
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              />
              <p className="text-[11px] text-slate-400 mt-1">Secondary co-borrower income</p>
              {errors.CoapplicantIncome && (
                <p className="text-xs text-rose-500 mt-1">{errors.CoapplicantIncome}</p>
              )}
            </div>

            {/* Loan Amount */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Loan Amount (in Thousands $) <span className="text-rose-500">*</span>
              </label>
              <input
                type="number"
                name="LoanAmount"
                value={formData.LoanAmount}
                onChange={handleChange}
                min="1"
                step="1"
                placeholder="e.g. 120 for $120,000"
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              />
              <p className="text-[11px] text-slate-400 mt-1">128 = $128,000</p>
              {errors.LoanAmount && <p className="text-xs text-rose-500 mt-1">{errors.LoanAmount}</p>}
            </div>

            {/* Loan Amount Term */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Loan Term (in Months) <span className="text-rose-500">*</span>
              </label>
              <select
                name="Loan_Amount_Term"
                value={formData.Loan_Amount_Term}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition"
              >
                <option value="360">360 Months (30 Years)</option>
                <option value="240">240 Months (20 Years)</option>
                <option value="180">180 Months (15 Years)</option>
                <option value="120">120 Months (10 Years)</option>
                <option value="84">84 Months (7 Years)</option>
                <option value="60">60 Months (5 Years)</option>
              </select>
              <p className="text-[11px] text-slate-400 mt-1">Repayment schedule duration</p>
              {errors.Loan_Amount_Term && (
                <p className="text-xs text-rose-500 mt-1">{errors.Loan_Amount_Term}</p>
              )}
            </div>

            {/* Credit History */}
            <div className="sm:col-span-2">
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Credit History Guidelines <span className="text-rose-500">*</span>
              </label>
              <select
                name="Credit_History"
                value={formData.Credit_History}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:bg-white outline-none transition font-medium"
              >
                <option value="1.0">1.0 – Meets Credit Guidelines (Favorable History)</option>
                <option value="0.0">0.0 – Does Not Meet Guidelines (Delinquency / Adverse History)</option>
              </select>
              <p className="text-[11px] text-slate-400 mt-1">
                Historical debt repayment compliance (key predictor in credit modeling)
              </p>
              {errors.Credit_History && (
                <p className="text-xs text-rose-500 mt-1">{errors.Credit_History}</p>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-6 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-end gap-3">
          <button
            type="button"
            onClick={handleReset}
            disabled={isLoading}
            className="w-full sm:w-auto px-5 py-3 text-sm font-medium text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 rounded-xl transition flex items-center justify-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset Form</span>
          </button>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full sm:w-auto px-8 py-3 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-60 rounded-xl shadow-lg shadow-blue-500/25 transition flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Evaluating Neighbors...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Predict Loan Approval</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

