"use client";

import React, { useState } from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import LoanForm from "@/components/LoanForm";
import PredictionResult from "@/components/PredictionResult";
import ModelInfo from "@/components/ModelInfo";
import Footer from "@/components/Footer";
import { PredictionResponse } from "@/types/loan";

export default function Home() {
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  return (
    <div className="flex-1 flex flex-col">
      <Navbar />

      <main className="flex-1">
        <Hero />

        {/* Prediction interactive section */}
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 -mt-6 sm:-mt-10 mb-20 relative z-20">
          <LoanForm
            onPredictionReceived={setPredictionResult}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            apiError={apiError}
            setApiError={setApiError}
          />

          {/* Dynamic Result Display */}
          <PredictionResult result={predictionResult} />
        </div>

        {/* About Model & Metrics Section */}
        <ModelInfo />
      </main>

      <Footer />
    </div>
  );
}

