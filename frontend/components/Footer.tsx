import React from "react";
import { ShieldCheck, Github, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-slate-900 text-slate-400 py-12 border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <span className="text-lg font-bold text-white">LoanWise AI</span>
            <span className="text-xs text-slate-500">|</span>
            <span className="text-xs text-slate-400">Machine Learning Capstone Project</span>
          </div>

          {/* Links */}
          <div className="flex items-center space-x-6 text-sm text-slate-400">
            <a href="#hero" className="hover:text-white transition">
              Home
            </a>
            <a href="#loan-form" className="hover:text-white transition">
              Prediction Tool
            </a>
            <a href="#model-info" className="hover:text-white transition">
              Model Details
            </a>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 pt-6 border-t border-slate-800/80 text-center md:text-left text-xs text-slate-500 leading-relaxed">
          <p>
            <strong>Disclaimer:</strong> This application demonstrates machine learning-based loan approval prediction
            using the K-Nearest Neighbors (KNN) algorithm. Its output is not an official financial decision or lending commitment.
            Financial institutions must adhere to regulatory compliance, fair lending practices (ECOA, FCRA), and human underwriter review.
          </p>
          <p className="mt-2 text-slate-600">
            &copy; {new Date().getFullYear()} LoanWise AI. Built for Academic and Machine Learning Demonstration.
          </p>
        </div>
      </div>
    </footer>
  );
}

