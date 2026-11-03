"use client";

import React, { useEffect, useState } from "react";
import {
  Brain,
  Layers,
  Scale,
  GitBranch,
  ShieldAlert,
  BarChart2,
  Table as TableIcon,
  CheckCircle,
  AlertOctagon,
} from "lucide-react";
import { ModelInfo as ModelInfoType } from "@/types/loan";
import { fetchModelInfo } from "@/lib/api";

export default function ModelInfo() {
  const [modelInfo, setModelInfo] = useState<ModelInfoType | null>(null);

  useEffect(() => {
    fetchModelInfo()
      .then((data) => setModelInfo(data))
      .catch((err) => console.warn("Using static metrics fallback:", err));
  }, []);

  // Verified actual values from the training run
  const selectedK = modelInfo?.selected_k ?? 19;
  const testAcc = (modelInfo?.test_accuracy ? modelInfo.test_accuracy * 100 : 85.37).toFixed(2);
  const cvAcc = (modelInfo?.best_cv_accuracy ? modelInfo.best_cv_accuracy * 100 : 79.83).toFixed(2);

  return (
    <section id="model-info" className="py-16 md:py-24 bg-slate-50 border-t border-slate-200">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-100 text-blue-800 text-xs font-semibold mb-3">
            <Brain className="w-4 h-4 text-blue-600" />
            <span>Machine Learning Architecture</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900">
            About the K-Nearest Neighbors (KNN) Model
          </h2>
          <p className="mt-3 text-base sm:text-lg text-slate-600">
            A comprehensive overview of our supervised classification pipeline, hyperparameter tuning methodology,
            and credit risk evaluation.
          </p>
        </div>

        {/* 4 Explanatory Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {/* Card 1: What is KNN */}
          <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition">
            <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center mb-4">
              <Layers className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-2">
              What is K-Nearest Neighbors (KNN)?
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              KNN is a non-parametric, instance-based supervised learning algorithm. Instead of learning an explicit mathematical
              equation during training, KNN stores historical loan applicant records and makes predictions at inference time
              by computing distances (e.g., Euclidean distance) between an incoming applicant and known training instances.
            </p>
          </div>

          {/* Card 2: How KNN Predicts */}
          <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-4">
              <GitBranch className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-2">
              How KNN Predicts Loan Approval
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              When an applicant submits their details, the algorithm maps their standardized features into multidimensional space.
              It finds the closest <strong>K = {selectedK}</strong> historical borrowers and tallies their recorded outcomes
              (Approved vs Rejected). The majority class determines the final recommendation, while class vote proportions define
              the confidence probability.
            </p>
          </div>

          {/* Card 3: Why Feature Scaling is Essential */}
          <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition">
            <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center mb-4">
              <Scale className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-2">
              Why Feature Scaling is Critical
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Distance metrics are extremely sensitive to numerical scales. For example, <code className="bg-slate-100 px-1 py-0.5 rounded text-xs">ApplicantIncome</code> ranges
              in thousands ($1,000–$80,000) while <code className="bg-slate-100 px-1 py-0.5 rounded text-xs">Credit_History</code> is bounded between 0 and 1. Without <code className="bg-slate-100 px-1 py-0.5 rounded text-xs">StandardScaler</code>,
              the income feature would artificially dominate distance calculations. Scaling maps all features to zero mean and unit variance.
            </p>
          </div>

          {/* Card 4: Why K=19 was Selected */}
          <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4">
              <BarChart2 className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-2">
              Optimal K Selection & Bias-Variance Tradeoff
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              A very small K (e.g., K=1 or 3) suffers from high variance and overfits to local noise or outliers. A very large K
              oversmooths decision boundaries and suffers from high bias. Using 5-Fold Stratified Cross-Validation across K=1 to 20,
              <strong> K = 19</strong> achieved the highest cross-validation stability (<strong>{cvAcc}%</strong>), ensuring robust generalization.
            </p>
          </div>
        </div>

        {/* Empirical Results & K Comparison Table */}
        <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-sm mb-12">
          <div className="flex items-center gap-2 mb-6">
            <TableIcon className="w-5 h-5 text-blue-600" />
            <h3 className="text-xl font-bold text-slate-900">
              Empirical Performance & K-Value Comparison
            </h3>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
            {/* Table */}
            <div className="overflow-x-auto lg:col-span-2">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50/70">
                    <th className="py-3 px-4 rounded-l-lg">Configuration</th>
                    <th className="py-3 px-4">K Value</th>
                    <th className="py-3 px-4">Distance Metric</th>
                    <th className="py-3 px-4 rounded-r-lg">Test Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  <tr>
                    <td className="py-3 px-4 font-medium">Experiment 1</td>
                    <td className="py-3 px-4">K = 3</td>
                    <td className="py-3 px-4">Minkowski (p=2)</td>
                    <td className="py-3 px-4 font-semibold text-slate-900">83.74%</td>
                  </tr>
                  <tr className="bg-blue-50/30">
                    <td className="py-3 px-4 font-medium">Default Baseline</td>
                    <td className="py-3 px-4">K = 5</td>
                    <td className="py-3 px-4">Minkowski (p=2)</td>
                    <td className="py-3 px-4 font-semibold text-slate-900">86.99%</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium">Experiment 2</td>
                    <td className="py-3 px-4">K = 7</td>
                    <td className="py-3 px-4">Minkowski (p=2)</td>
                    <td className="py-3 px-4 font-semibold text-slate-900">84.55%</td>
                  </tr>
                  <tr className="bg-emerald-50/40 font-bold text-emerald-900">
                    <td className="py-3 px-4 rounded-l-lg flex items-center gap-1.5">
                      <CheckCircle className="w-4 h-4 text-emerald-600" />
                      <span>Optimized (GridSearchCV)</span>
                    </td>
                    <td className="py-3 px-4">K = 19</td>
                    <td className="py-3 px-4">Euclidean (Uniform)</td>
                    <td className="py-3 px-4 rounded-r-lg text-emerald-700">{testAcc}% (CV: {cvAcc}%)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Metric Summary Card */}
            <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 text-center">
              <span className="text-xs uppercase font-bold tracking-wider text-slate-400">
                Evaluation Test Split
              </span>
              <p className="text-3xl font-extrabold text-slate-900 mt-1">123 Applicants</p>
              <p className="text-xs text-slate-500 mt-1">20% Stratified Holdout</p>
              <div className="mt-4 pt-4 border-t border-slate-200/80 grid grid-cols-2 gap-2 text-left">
                <div>
                  <span className="text-[11px] text-slate-400 font-semibold block">Approved Precision</span>
                  <span className="text-sm font-bold text-slate-800">83.2%</span>
                </div>
                <div>
                  <span className="text-[11px] text-slate-400 font-semibold block">Approved Recall</span>
                  <span className="text-sm font-bold text-slate-800">98.8%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Analysis: False Approvals vs False Rejections */}
        <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <ShieldAlert className="w-5 h-5 text-rose-600" />
            <h3 className="text-xl font-bold text-slate-900">
              Financial Risk Assessment: False Approvals vs False Rejections
            </h3>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed mb-6">
            In credit underwriting, high overall accuracy alone is insufficient. Machine learning models must be analyzed through
            the lens of asymmetric cost: approving a delinquent borrower incurs substantially greater losses than erroneously turning away a creditworthy customer.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* False Approval Card (FP) */}
            <div className="p-5 rounded-xl border border-rose-200 bg-rose-50/50">
              <div className="flex items-center gap-2 text-rose-700 font-bold text-sm mb-2">
                <AlertOctagon className="w-4 h-4" />
                <span>False Approval (Type I Error / False Positive): 17 Cases</span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                <strong>Financial Impact: Severe Credit Default Risk.</strong> The model predicted &ldquo;Approved&rdquo; for an applicant who historically defaulted or was rejected.
                These loans lead to principal loss, loan recovery expenses, and increased capital provisioning requirements for the financial institution.
              </p>
            </div>

            {/* False Rejection Card (FN) */}
            <div className="p-5 rounded-xl border border-amber-200 bg-amber-50/50">
              <div className="flex items-center gap-2 text-amber-700 font-bold text-sm mb-2">
                <ShieldAlert className="w-4 h-4" />
                <span>False Rejection (Type II Error / False Negative): 1 Case</span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                <strong>Financial Impact: Lost Net Interest Margin & Customer Friction.</strong> The model rejected an applicant who was actually creditworthy.
                With only 1 false negative on the test set, our optimized model achieved an outstanding <strong>98.8% Recall</strong> on legitimate borrowers.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

