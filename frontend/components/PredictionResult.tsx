"use client";

import React from "react";
import {
  CheckCircle2,
  XCircle,
  TrendingUp,
  Percent,
  FileText,
  AlertTriangle,
  Info,
  DollarSign,
  Calendar,
  Building,
  GraduationCap,
} from "lucide-react";
import { PredictionResponse } from "@/types/loan";

interface PredictionResultProps {
  result: PredictionResponse | null;
}

export default function PredictionResult({ result }: PredictionResultProps) {
  if (!result) return null;

  const isApproved = result.prediction === "Approved";
  const confidencePct = (result.confidence * 100).toFixed(1);
  const approvedPct = (result.class_probabilities.Approved * 100).toFixed(1);
  const rejectedPct = (result.class_probabilities.Rejected * 100).toFixed(1);

  return (
    <section id="prediction-result" className="mt-12 scroll-mt-24">
      <div
        className={`rounded-2xl border p-6 sm:p-8 md:p-10 shadow-xl transition-all duration-300 ${
          isApproved
            ? "bg-gradient-to-br from-emerald-500/5 via-white to-white border-emerald-300 shadow-emerald-500/10"
            : "bg-gradient-to-br from-rose-500/5 via-white to-white border-rose-300 shadow-rose-500/10"
        }`}
      >
        {/* Status Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-100 gap-4">
          <div className="flex items-center space-x-4">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center text-white shadow-lg ${
                isApproved
                  ? "bg-gradient-to-tr from-emerald-600 to-teal-500 shadow-emerald-500/25"
                  : "bg-gradient-to-tr from-rose-600 to-red-500 shadow-rose-500/25"
              }`}
            >
              {isApproved ? (
                <CheckCircle2 className="w-8 h-8" />
              ) : (
                <XCircle className="w-8 h-8" />
              )}
            </div>
            <div>
              <span
                className={`text-xs font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full ${
                  isApproved
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-rose-100 text-rose-800"
                }`}
              >
                Model Decision Code: {result.prediction_code}
              </span>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1">
                Loan Application:{" "}
                <span
                  className={isApproved ? "text-emerald-600" : "text-rose-600"}
                >
                  {result.prediction}
                </span>
              </h2>
            </div>
          </div>

          {/* Confidence Badge */}
          <div className="flex flex-col sm:items-end">
            <div className="text-xs text-slate-500 font-medium">Model Confidence</div>
            <div className="text-2xl font-black text-slate-900 flex items-center gap-1">
              <span>{confidencePct}%</span>
              <Percent className="w-5 h-5 text-blue-600" />
            </div>
          </div>
        </div>

        {/* Narrative & Probability Bars */}
        <div className="py-6 grid grid-cols-1 lg:grid-cols-2 gap-8 items-center border-b border-slate-100">
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
              Algorithmic Assessment
            </h4>
            <p className="text-slate-700 text-base leading-relaxed">
              {result.message}
            </p>
            <div className="mt-4 p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 text-xs text-slate-600 flex items-start gap-2.5">
              <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <span>
                Based on Euclidean proximity to the K = 19 nearest historical training profiles,
                this applicant exhibits {isApproved ? "favorable solvency markers" : "heightened delinquency risk"}.
              </span>
            </div>
          </div>

          {/* Class Probabilities Distribution */}
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200/80">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-4 flex items-center justify-between">
              <span>Neighbor Class Probabilities (K = 19)</span>
              <TrendingUp className="w-4 h-4 text-slate-400" />
            </h4>

            {/* Approved bar */}
            <div className="mb-4">
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-emerald-700">Approved (Y)</span>
                <span className="text-slate-700">{approvedPct}%</span>
              </div>
              <div className="h-3 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                  style={{ width: `${approvedPct}%` }}
                />
              </div>
            </div>

            {/* Rejected bar */}
            <div>
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-rose-700">Rejected (N)</span>
                <span className="text-slate-700">{rejectedPct}%</span>
              </div>
              <div className="h-3 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-rose-500 rounded-full transition-all duration-500"
                  style={{ width: `${rejectedPct}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Applicant Summary Cards */}
        <div className="pt-6">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-1.5">
            <FileText className="w-4 h-4 text-blue-600" />
            <span>Submitted Profile Overview</span>
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 bg-white border border-slate-200 rounded-xl">
              <span className="text-[11px] font-semibold text-slate-400 block">Total Income</span>
              <span className="text-sm font-bold text-slate-800">
                ${result.applicant_summary.total_income?.toLocaleString()}
              </span>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-xl">
              <span className="text-[11px] font-semibold text-slate-400 block">Loan Requested</span>
              <span className="text-sm font-bold text-slate-800">
                ${(result.applicant_summary.loan_amount_k * 1000)?.toLocaleString()}
              </span>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-xl">
              <span className="text-[11px] font-semibold text-slate-400 block">Credit History</span>
              <span
                className={`text-sm font-bold ${
                  result.applicant_summary.credit_history.includes("Meets")
                    ? "text-emerald-600"
                    : "text-rose-600"
                }`}
              >
                {result.applicant_summary.credit_history}
              </span>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-xl">
              <span className="text-[11px] font-semibold text-slate-400 block">Area & Education</span>
              <span className="text-sm font-bold text-slate-800">
                {result.applicant_summary.property_area} • {result.applicant_summary.education}
              </span>
            </div>
          </div>
        </div>

        {/* Disclaimer footer */}
        <div className="mt-8 pt-4 border-t border-slate-100 flex items-start gap-2.5 text-xs text-slate-500">
          <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
          <p>{result.disclaimer}</p>
        </div>
      </div>
    </section>
  );
}

