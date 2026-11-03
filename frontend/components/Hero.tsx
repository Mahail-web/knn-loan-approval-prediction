"use client";

import React from "react";
import { ArrowDown, Cpu, Sparkles, TrendingUp, BarChart3 } from "lucide-react";

export default function Hero() {
  return (
    <section id="hero" className="relative overflow-hidden pt-12 pb-16 md:pt-20 md:pb-24 bg-gradient-to-b from-blue-50/50 via-slate-50 to-white">
      {/* Background blur shapes */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-blue-400/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -top-10 right-10 w-72 h-72 bg-indigo-300/10 rounded-full blur-2xl pointer-events-none" />

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
        {/* Badge */}
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-blue-100/70 text-blue-800 text-xs sm:text-sm font-semibold mb-6 border border-blue-200">
          <Sparkles className="w-4 h-4 text-blue-600" />
          <span>Machine Learning Capstone – Supervised KNN Algorithm</span>
        </div>

        {/* Main Heading */}
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-slate-900 tracking-tight leading-tight">
          Smart Loan Approval Prediction
        </h1>

        {/* Supporting text */}
        <p className="mt-5 max-w-3xl mx-auto text-base sm:text-lg md:text-xl text-slate-600 leading-relaxed">
          Analyze applicant information and explore loan approval predictions powered by Machine Learning.
          Leveraging an optimized K-Nearest Neighbors classifier trained on historical credit profiles.
        </p>

        {/* CTA Button */}
        <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="#loan-form"
            className="w-full sm:w-auto inline-flex items-center justify-center px-8 py-3.5 text-base font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 rounded-xl shadow-lg shadow-blue-500/25 transition-all duration-200 group"
          >
            <span>Predict Loan Approval</span>
            <ArrowDown className="w-4 h-4 ml-2 group-hover:translate-y-0.5 transition-transform" />
          </a>
          <a
            href="#model-info"
            className="w-full sm:w-auto inline-flex items-center justify-center px-6 py-3.5 text-base font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-300 rounded-xl shadow-sm transition-all duration-200"
          >
            <span>Explore Model Metrics</span>
          </a>
        </div>

        {/* Feature stats cards */}
        <div className="mt-14 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto text-left">
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center space-x-2 text-blue-600 mb-1">
              <Cpu className="w-4 h-4" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Algorithm</span>
            </div>
            <p className="text-xl font-bold text-slate-900">K-Nearest Neighbors</p>
            <p className="text-xs text-slate-500 mt-0.5">Optimized K = 19</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center space-x-2 text-indigo-600 mb-1">
              <TrendingUp className="w-4 h-4" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Test Accuracy</span>
            </div>
            <p className="text-xl font-bold text-slate-900">85.37%</p>
            <p className="text-xs text-slate-500 mt-0.5">123 Untouched Samples</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center space-x-2 text-emerald-600 mb-1">
              <BarChart3 className="w-4 h-4" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">5-Fold CV Score</span>
            </div>
            <p className="text-xl font-bold text-slate-900">79.83%</p>
            <p className="text-xs text-slate-500 mt-0.5">GridSearchCV Fold Avg</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center space-x-2 text-amber-600 mb-1">
              <Sparkles className="w-4 h-4" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">StandardScaler</span>
            </div>
            <p className="text-xl font-bold text-slate-900">Zero Leakage</p>
            <p className="text-xs text-slate-500 mt-0.5">Scikit-learn Pipeline</p>
          </div>
        </div>
      </div>
    </section>
  );
}

