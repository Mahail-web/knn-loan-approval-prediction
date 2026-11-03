"""
Visualization module for Loan Approval Prediction using KNN.
Generates 7 required professional visualizations using Matplotlib & Seaborn.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set professional theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "figure.dpi": 150,
})


def plot_loan_approval_distribution(df: pd.DataFrame, output_dir: str):
    """1. Loan approval distribution (Approved vs Rejected)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Loan_Status"].value_counts()
    percentages = df["Loan_Status"].value_counts(normalize=True) * 100
    labels = ["Approved (Y)", "Rejected (N)"] if counts.index[0] == "Y" else ["Rejected (N)", "Approved (Y)"]
    
    colors = ["#2563EB", "#EF4444"]
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    
    for bar, pct, count in zip(bars, percentages.values, counts.values):
        height = bar.get_height()
        ax.annotate(
            f"{count} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_title("Target Variable Distribution: Loan Approval Status", pad=15, fontweight="bold")
    ax.set_xlabel("Loan Status", labelpad=10)
    ax.set_ylabel("Number of Applicants", labelpad=10)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "loan_approval_distribution.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_missing_values(df: pd.DataFrame, output_dir: str):
    """2. Missing values visualization across all features."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    if not missing.empty:
        bars = ax.barh(missing.index, missing.values, color="#F59E0B", edgecolor="black")
        for bar in bars:
            width = bar.get_width()
            pct = (width / len(df)) * 100
            ax.annotate(
                f"{int(width)} ({pct:.1f}%)",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=10,
            )
        ax.set_xlim(0, max(missing.values) * 1.25)
    else:
        ax.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center", fontsize=14)

    ax.set_title("Missing Values Count and Percentage per Feature", pad=15, fontweight="bold")
    ax.set_xlabel("Number of Missing Values", labelpad=10)
    ax.set_ylabel("Dataset Features", labelpad=10)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "missing_values_visualization.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_numerical_distributions(df: pd.DataFrame, output_dir: str):
    """3. Numerical feature distributions (Histograms + KDEs)."""
    numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color="#3B82F6", edgecolor="black", alpha=0.6)
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="#059669", linestyle="-", linewidth=1.5, label=f"Median: {median_val:.1f}")
        ax.set_title(f"Distribution of {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density / Count")
        ax.legend(frameon=True, loc="upper right")

    plt.suptitle("Distributions of Numerical Features in Loan Dataset", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "numerical_feature_distributions.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """4. Correlation heatmap for numerical features and binary target."""
    temp_df = df.copy()
    if "Loan_Status" in temp_df.columns:
        temp_df["Loan_Status_Num"] = temp_df["Loan_Status"].map({"Y": 1, "N": 0})
    
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    if "Loan_Status_Num" in temp_df.columns:
        num_cols.append("Loan_Status_Num")
        
    corr = temp_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Numerical Features & Target", pad=15, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_k_value_comparison(k_results: dict, cv_results: dict, output_dir: str):
    """
    5. Accuracy comparison for K = 3, 5, and 7, alongside the complete K range (1 to 20).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart for K=3, 5, 7 test accuracy
    k_keys = [str(k) for k in [3, 5, 7] if k in k_results]
    k_vals = [k_results[k] * 100 for k in [3, 5, 7] if k in k_results]
    colors = ["#60A5FA", "#3B82F6", "#1D4ED8"]
    bars = ax1.bar(k_keys, k_vals, color=colors, edgecolor="black", width=0.45)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax1.set_title("Test Accuracy for Specific K Values (3, 5, 7)", fontweight="bold")
    ax1.set_xlabel("Number of Neighbors (K)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_ylim(60, 90)

    # Line plot for full K range CV tuning curve
    if cv_results:
        k_list = cv_results.get("k_list", [])
        mean_scores = [s * 100 for s in cv_results.get("mean_cv_scores", [])]
        best_k = cv_results.get("best_k", 5)
        best_score = cv_results.get("best_score", 0) * 100

        ax2.plot(k_list, mean_scores, marker="o", color="#2563EB", linewidth=2, label="5-Fold Mean CV Accuracy")
        ax2.axvline(best_k, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Best K = {best_k} ({best_score:.2f}%)")
        ax2.scatter([best_k], [best_score], color="#DC2626", s=100, zorder=5)
        ax2.set_title("Cross-Validation Accuracy vs K (1 to 20)", fontweight="bold")
        ax2.set_xlabel("Number of Neighbors (K)")
        ax2.set_ylabel("CV Accuracy (%)")
        ax2.set_xticks(range(1, max(k_list) + 1, 2))
        ax2.legend(frameon=True)

    plt.suptitle("Hyperparameter Tuning: K-Nearest Neighbors Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "k_value_accuracy_comparison.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_confusion_matrix(cm_array, model_name: str, filename: str, output_dir: str):
    """
    6 & 7. Confusion Matrix plot with clearly labeled axes:
    Actual Class vs Predicted Class, with labels Approved and Rejected.
    """
    # Matrix format:
    # Row 0: Actual Rejected (0) -> [TN (Rejected), FP (Approved)]
    # Row 1: Actual Approved (1) -> [FN (Rejected), TP (Approved)]
    cm_display = np.array(cm_array)
    labels = ["Rejected", "Approved"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    annot_matrix = np.array([
        [f"TN\n{cm_display[0,0]}", f"FP (False Approval)\n{cm_display[0,1]}"],
        [f"FN (False Rejection)\n{cm_display[1,0]}", f"TP\n{cm_display[1,1]}"]
    ])

    sns.heatmap(
        cm_display,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1.5,
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_title(f"Confusion Matrix: {model_name}", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Class", labelpad=10, fontweight="bold")
    ax.set_ylabel("Actual Class", labelpad=10, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

