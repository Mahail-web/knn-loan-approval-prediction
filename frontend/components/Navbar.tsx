"use client";

import React, { useEffect, useState } from "react";
import { ShieldCheck, Activity, Cpu } from "lucide-react";
import { fetchHealth } from "@/lib/api";

export default function Navbar() {
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((data) => setIsBackendHealthy(data.model_loaded))
      .catch(() => setIsBackendHealthy(false));
  }, []);

  return (
    <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-700 to-indigo-700 bg-clip-text text-transparent">
                LoanWise AI
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs px-2 py-0.5 font-medium bg-blue-50 text-blue-700 border border-blue-200 rounded-full">
                KNN v1.0
              </span>
            </div>
          </div>

          {/* Nav Links */}
          <div className="hidden md:flex items-center space-x-8 text-sm font-medium text-slate-600">
            <a href="#hero" className="hover:text-blue-600 transition-colors">
              Home
            </a>
            <a href="#loan-form" className="hover:text-blue-600 transition-colors">
              Loan Prediction
            </a>
            <a href="#model-info" className="hover:text-blue-600 transition-colors">
              About the Model
            </a>
          </div>

          {/* Backend Status indicator */}
          <div className="flex items-center space-x-2 text-xs">
            <div
              className={`flex items-center px-3 py-1.5 rounded-full font-medium ${
                isBackendHealthy === true
                  ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                  : isBackendHealthy === false
                  ? "bg-rose-50 text-rose-700 border border-rose-200"
                  : "bg-slate-100 text-slate-600 border border-slate-200"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full mr-2 ${
                  isBackendHealthy === true
                    ? "bg-emerald-500 animate-pulse"
                    : isBackendHealthy === false
                    ? "bg-rose-500"
                    : "bg-slate-400"
                }`}
              />
              {isBackendHealthy === true
                ? "API Connected"
                : isBackendHealthy === false
                ? "API Offline"
                : "Checking API..."}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

