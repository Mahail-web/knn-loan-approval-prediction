import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LoanWise AI – Loan Approval Prediction System",
  description:
    "End-to-end Machine Learning Capstone application predicting loan approvals with K-Nearest Neighbors (KNN), FastAPI, and Next.js.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-slate-50 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}

