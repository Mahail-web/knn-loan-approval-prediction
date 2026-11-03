# LoanWise AI: Loan Approval Prediction Using K-Nearest Neighbors (KNN)

**Capstone Machine Learning Project**  
*A complete end-to-end Machine Learning, REST API, and Full-Stack Next.js Application*

---

## 1. Project Introduction

**LoanWise AI** is a production-grade machine learning application designed to automate and augment credit risk evaluation for personal and mortgage loan approvals. Loan underwriting in retail banking requires evaluating multiple applicant dimensions—including applicant income, co-borrower financial strength, requested loan size, loan amortization term, property geography, and historical credit behavior.

By utilizing the **K-Nearest Neighbors (KNN)** classification algorithm embedded in a strict, data-leakage-free Scikit-learn Pipeline with automated imputation, one-hot encoding, and feature standardization, this project demonstrates the complete lifecycle of applied machine learning: from raw exploratory data analysis and rigorous hyperparameter tuning via cross-validation, to RESTful model serving with FastAPI and a modern interactive interface built with Next.js and Tailwind CSS.

---

## 2. Project Objectives

- **Complete ML Pipeline**: Implement modular data ingestion, exploratory data analysis (EDA), automated imputation, encoding, and scaling.
- **Strict Leakage Prevention**: Ensure feature transformers (e.g., `StandardScaler`, `SimpleImputer`) are fitted strictly on training folds within a unified `ColumnTransformer` and `Pipeline`.
- **Systematic Hyperparameter Optimization**: Evaluate default parameters against explicit values ($K = 3, 5, 7$) and tune $K \in [1, 20]$ with distance metrics and weighting schemes using 5-fold Stratified Cross-Validation (`GridSearchCV`).
- **Asymmetric Risk Assessment**: Analyze model trade-offs between **False Approvals** (credit default risk / False Positives) and **False Rejections** (lost interest margin / False Negatives).
- **Production REST API**: Deploy a FastAPI microservice exposing model metadata, health checks, and JSON-based predictions with class probabilities (`predict_proba`).
- **Fintech Web Application**: Deliver a responsive Next.js frontend with live input validation, quick preset scenarios, dynamic probability distributions, and transparent metric explainability.

---

## 3. Dataset Source

The dataset used is the benchmark **Loan Prediction Problem Dataset** available on Kaggle:
- **Kaggle URL**: [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Primary Data File**: `train.csv` (stored in `ml/data/train.csv`)
- **Sample Count**: 614 applicant records
- **Feature Count**: 13 columns (1 identifier `Loan_ID`, 11 predictive features, 1 target `Loan_Status`)
- **Target Distribution**:
  - **Approved (`Y`)**: 422 records (68.73%)
  - **Rejected (`N`)**: 192 records (31.27%)

---

## 4. Technologies Used

| Domain | Technology / Library | Purpose |
|---|---|---|
| **Language** | Python 3.11 | Core pipeline, model training, and API runtime |
| **Data Processing** | Pandas 2.2+, NumPy 1.26+ | Dataframe manipulation, statistical arrays |
| **Machine Learning** | Scikit-learn 1.4+ | `Pipeline`, `ColumnTransformer`, `KNeighborsClassifier`, `GridSearchCV` |
| **Model Persistence**| Joblib 1.3+ | Pipeline serialization (`best_knn_model.joblib`) |
| **Visualizations** | Matplotlib 3.8+, Seaborn 0.13+ | Statistical distribution and performance plots |
| **Notebook Analysis**| Jupyter / `nbformat` | Interactive research notebook (`loan_approval_analysis.ipynb`) |
| **Backend Framework**| FastAPI, Uvicorn, Pydantic v2 | High-performance asynchronous REST API |
| **Frontend UI** | Next.js 14 (App Router), React 18 | SSR and interactive client web interface |
| **Language (Web)** | TypeScript 5.4+ | Type safety across client components and API types |
| **Styling** | Tailwind CSS 3.4+ | Clean fintech design system |
| **Icons** | Lucide React | Modern iconography |

---

## 5. Machine Learning Methodology

```
[ Raw CSV Data: train.csv ]
              │
              ▼
[ Feature Separation & Target Mapping ]
  • Remove Loan_ID
  • Map Loan_Status: Y -> 1 (Approved), N -> 0 (Rejected)
              │
              ▼
[ Stratified 80/20 Train-Test Split ]
  • Training Set: 491 samples (Stratified)
  • Testing Set:  123 samples (Stratified holdout)
              │
              ▼
[ Scikit-Learn ColumnTransformer ]
  ├─ Numerical Pipeline:   SimpleImputer(median) ──> StandardScaler()
  └─ Categorical Pipeline: SimpleImputer(mode)   ──> OneHotEncoder(ignore_unknown)
              │
              ▼
[ 5-Fold Stratified Cross-Validation (GridSearchCV) ]
  • Explores K in [1, 20], weights in [uniform, distance], metrics in [euclidean, manhattan]
  • Zero Data Leakage: Scaler & Imputer fitted strictly within each CV fold
              │
              ▼
[ Optimal Estimator Selection ]
  • Best Model: K = 19, metric = euclidean, weights = uniform
              │
              ▼
[ Untouched Test Set Evaluation ]
  • Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Financial Risk
              │
              ▼
[ Serialization: best_knn_model.joblib & model_config.json ]
```

---

## 6. Data Preprocessing Steps

1. **Identifier Elimination**: Dropped `Loan_ID` because unique tracking strings possess zero generalization power.
2. **Missing Value Imputation**:
   - **Numerical Features** (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`): Imputed with the feature's **median**, which is robust to extreme outliers common in income and debt distributions.
   - **Categorical Features** (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`): Imputed with the **mode** (most frequent class).
3. **Categorical Encoding**:
   - Nominal categories are transformed into numerical vectors using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - Handling unknown values safely ensures that unexpected categories in incoming API requests do not trigger runtime crashes.
4. **Target Encoding**:
   - `Loan_Status`: `Y` mapped to `1` (Approved), `N` mapped to `0` (Rejected).
5. **Feature Standardization**:
   - Features standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
   - Fit strictly on training folds to prevent data leakage.

---

## 7. K-Nearest Neighbors (KNN) Algorithm Explanation

K-Nearest Neighbors is an intuitive yet powerful non-parametric instance-based classifier.

### Mathematical Formulation
Given a query applicant vector $\mathbf{x} \in \mathbb{R}^d$ and training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$:
1. **Distance Metric (Euclidean)**:
   $$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{d} (x_j - x_{i,j})^2}$$
2. **Neighborhood Selection**: The algorithm selects the index set $\mathcal{N}_K(\mathbf{x})$ containing the $K$ smallest distances $d(\mathbf{x}, \mathbf{x}_i)$.
3. **Classification Rule**:
   $$\hat{y} = \arg\max_{c \in \{0, 1\}} \sum_{i \in \mathcal{N}_K(\mathbf{x})} \mathbb{I}(y_i = c)$$
4. **Class Probability**:
   $$P(y = \text{Approved} \mid \mathbf{x}) = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\mathbf{x})} \mathbb{I}(y_i = 1)$$

### Why Feature Scaling is Essential for KNN
Because distances aggregate squared differences across dimensions, unscaled features with large raw magnitudes (such as `ApplicantIncome` ranging up to \$81,000) completely drown out features with small ranges (such as `Credit_History` $\in \{0, 1\}$). Without `StandardScaler`, `Credit_History`—the single strongest empirical predictor of loan repayment—would contribute less than 0.001% to distance calculations.

---

## 8. Hyperparameter Tuning Methodology

We employed `GridSearchCV` combined with `StratifiedKFold` ($k=5$ splits, `random_state=42`) to systematically optimize:
- **`n_neighbors`**: Evaluated all integers from 1 to 20.
- **`weights`**: `['uniform', 'distance']`
- **`metric`**: `['euclidean', 'manhattan']`

### Grid Search Outcomes
- **Best Hyperparameters**:
  - `n_neighbors`: **19**
  - `metric`: **`euclidean`**
  - `weights`: **`uniform`**
- **Best 5-Fold Cross-Validation Accuracy**: **79.83%**

---

## 9. Model Evaluation Results

### 9.1 K-Value Experiment Comparison (Untouched Test Set, 123 Samples)

| Configuration | K Value | Metric | Test Accuracy | Status |
|---|---|---|---|---|
| Experiment 1 | $K = 3$ | Minkowski ($p=2$) | **83.74%** | Baseline variant |
| Default Model | $K = 5$ | Minkowski ($p=2$) | **86.99%** | Default Scikit-learn |
| Experiment 2 | $K = 7$ | Minkowski ($p=2$) | **84.55%** | Baseline variant |
| **Tuned Model** | **$K = 19$** | **Euclidean (Uniform)** | **85.37%** | **Optimal CV Model** (CV: 79.83%) |

> **Discussion on Number of Neighbors ($K$)**: Small values of $K$ ($K=3$) exhibit high variance and are susceptible to localized label noise in personal credit records. Larger neighborhoods ($K=19$) provide smoother decision boundaries and superior cross-validation generalization across diverse folds.

### 9.2 Tuned Model Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Rejected (`0`)** | 0.95 | 0.55 | 0.70 | 38 |
| **Approved (`1`)** | 0.83 | 0.99 | 0.90 | 85 |
| **Accuracy** | | | **0.8537** | 123 |
| **Macro Average** | 0.89 | 0.77 | 0.80 | 123 |
| **Weighted Average** | 0.87 | 0.85 | 0.84 | 123 |

### 9.3 Financial Credit Risk & Error Analysis

The confusion matrix on the 123 test applicants ($N=38$ actual rejected, $N=85$ actual approved):

```
                       Predicted Class
                    Rejected      Approved
Actual  Rejected      21 (TN)       17 (FP)
Class   Approved       1 (FN)       84 (TP)
```

1. **False Approvals (False Positives = 17)**:
   - **Risk**: Severe financial default risk. The model predicted loan approval for applicants who historically defaulted or failed underwriting.
   - **Mitigation**: In institutional production, high loan-to-value applications close to the 50% probability threshold must be routed to secondary underwriter review.
2. **False Rejections (False Negatives = 1)**:
   - **Risk**: Lost net interest margin and customer dissatisfaction.
   - **Performance**: The model achieved an exceptional **98.8% Recall (84 / 85)** on creditworthy borrowers, turning away almost zero legitimate clients.

---

## 10. Project Folder Structure

```
loan-approval-prediction/
├── ml/
│   ├── data/
│   │   └── train.csv                     # Raw Kaggle Loan Prediction dataset
│   ├── notebooks/
│   │   ├── loan_approval_analysis.ipynb  # Comprehensive Jupyter Analysis
│   │   └── generate_notebook.py          # Notebook compilation script
│   ├── src/
│   │   ├── preprocessing.py              # ColumnTransformer & imputation pipelines
│   │   ├── train.py                      # Model training & GridSearchCV logic
│   │   ├── evaluate.py                   # Evaluation metrics & risk assessment
│   │   ├── visualize.py                  # 7 publication-quality Matplotlib/Seaborn plots
│   │   ├── save_model.py                 # Pipeline serialization & config persistence
│   │   └── run_pipeline.py               # Master orchestration script
│   ├── models/
│   │   ├── best_knn_model.joblib         # Serialized Scikit-learn Pipeline
│   │   └── model_config.json             # Model metadata & empirical scores
│   ├── visualizations/                   # High-res generated figures
│   │   ├── loan_approval_distribution.png
│   │   ├── missing_values_visualization.png
│   │   ├── numerical_feature_distributions.png
│   │   ├── correlation_heatmap.png
│   │   ├── k_value_accuracy_comparison.png
│   │   ├── confusion_matrix_default_knn.png
│   │   └── confusion_matrix_optimized_knn.png
│   └── requirements.txt                  # Python ML dependencies
│
├── backend/
│   ├── main.py                           # FastAPI application & route controllers
│   ├── model.py                          # ModelManager singleton & inference runner
│   ├── schemas.py                        # Pydantic v2 schemas & request validators
│   ├── best_knn_model.joblib             # Production model pipeline (synced)
│   ├── model_config.json                 # Production model config (synced)
│   ├── requirements.txt                  # Backend dependencies (FastAPI, uvicorn)
│   └── .env.example                      # Environment variables template
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx                    # Next.js Root Layout & metadata
│   │   ├── page.tsx                      # Main interactive application page
│   │   └── globals.css                   # Tailwind styles & theme variables
│   ├── components/
│   │   ├── Navbar.tsx                    # Brand navigation & live API status indicator
│   │   ├── Hero.tsx                      # Hero banner with feature statistics
│   │   ├── LoanForm.tsx                  # 11-field credit application form with test presets
│   │   ├── PredictionResult.tsx          # Dynamic approval card & probability meters
│   │   ├── ModelInfo.tsx                 # Architecture explainers & empirical tables
│   │   └── Footer.tsx                    # Legal and educational disclaimer
│   ├── lib/
│   │   ├── api.ts                        # Typed API fetchers
│   │   └── validation.ts                 # Client-side input validation rules
│   ├── types/
│   │   └── loan.ts                       # TypeScript interfaces for applicants & responses
│   ├── package.json                      # Next.js scripts & dependencies
│   ├── tailwind.config.js                # Tailwind CSS configuration
│   ├── tsconfig.json                     # TypeScript configuration
│   └── .env.local.example                # Frontend environment template
│
├── README.md                             # Comprehensive project documentation
└── .gitignore                            # Version control exclusion rules
```

---

## 11. Installation Instructions

### Prerequisites
- **Python**: Version 3.10+ (Python 3.11 recommended)
- **Node.js**: Version 18.0+ (Node v20/v24 tested)
- **Git**

### Step 1: Clone Repository & Set Up Virtual Environment

```bash
git clone <repository-url>
cd "d:\Datacrumbs assignments\KNN"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# Install ML and backend dependencies
pip install -r ml/requirements.txt
pip install -r backend/requirements.txt
```

### Step 2: Run Machine Learning Pipeline (Generates Model & Visualizations)

```bash
python ml/src/run_pipeline.py
```
*This command executes exploratory analysis, trains default and experimental models, executes 5-fold cross-validation, saves all 7 visualizations to `ml/visualizations/`, and serializes the best pipeline to `ml/models/best_knn_model.joblib` and `backend/best_knn_model.joblib`.*

---

## 12. Backend Execution Instructions

Navigate to the `backend/` directory and launch the Uvicorn server:

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:
- **API Base**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Alternative ReDoc**: `http://localhost:8000/redoc`

---

## 13. Frontend Execution Instructions

Open a new terminal window, navigate to `frontend/`, configure environment variables, and start the development server:

```bash
cd frontend
npm install
npm run dev
```

The application will start on:
- **Web Interface**: `http://localhost:3000`

---

## 14. API Endpoint Documentation

### 1. `GET /`
Returns general application information and index of available endpoints.

### 2. `GET /health`
Returns system health, timestamp, and verification that the model pipeline is loaded in memory.

### 3. `GET /model-info`
Returns full metadata, selected $K$, hyperparameters, cross-validation scores, and classification metrics.

### 4. `POST /predict`
Processes loan applicant data through the Scikit-learn Pipeline and returns the binary decision and probability distribution.

---

## 15. Example API Request and Response

### Example Request (`POST /predict`)

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Gender": "Male",
       "Married": "Yes",
       "Dependents": "2",
       "Education": "Graduate",
       "Self_Employed": "No",
       "ApplicantIncome": 6000.0,
       "CoapplicantIncome": 2500.0,
       "LoanAmount": 130.0,
       "Loan_Amount_Term": 360.0,
       "Credit_History": 1.0,
       "Property_Area": "Semiurban"
     }'
```

### Example Response

```json
{
  "success": true,
  "prediction": "Approved",
  "prediction_code": "Y",
  "message": "The model predicts that this loan application will be approved based on neighbor similarity patterns.",
  "confidence": 0.9474,
  "class_probabilities": {
    "Approved": 0.9474,
    "Rejected": 0.0526
  },
  "applicant_summary": {
    "applicant_income": 6000.0,
    "coapplicant_income": 2500.0,
    "total_income": 8500.0,
    "loan_amount_k": 130.0,
    "term_months": 360.0,
    "credit_history": "Meets Guidelines (1.0)",
    "property_area": "Semiurban",
    "education": "Graduate"
  },
  "disclaimer": "This application demonstrates machine learning-based loan approval prediction using K-Nearest Neighbors. Its output is an algorithmic estimation and does not constitute a legal, formal, or binding financial commitment."
}
```

---

## 16. Known Limitations

1. **Inference Latency on Large Datasets ($O(N \cdot D)$)**: KNN is a lazy learner that stores instances and computes distance to training examples at inference time. While instantaneous for 614 rows, inference on millions of records requires indexing trees (BallTree/KDTree) or approximate nearest neighbors (e.g., HNSW).
2. **High Dimensionality Sensitivity**: As features increase, the distance between data points becomes equidistant (curse of dimensionality), reducing the discriminative power of Euclidean distance.
3. **Imbalanced Representation of Rejected Borrowers**: In retail credit datasets, approved loans (68.7%) outnumber rejections (31.3%), leading to higher sensitivity toward approvals than rejections.
4. **Historical Bias**: Machine learning models can mirror historical bias present in training records; ethical deployment requires strict fairness auditing across demographic attributes.

---

## 17. Future Improvements

- **Tree-Based Ensembles**: Benchmark against gradient boosting frameworks (XGBoost, LightGBM, CatBoost) to capture non-linear feature interactions without distance calculations.
- **Cost-Sensitive Loss Optimization**: Train with asymmetric misclassification penalties where False Approvals are penalized $5\times$ more heavily than False Rejections.
- **Explainable AI (SHAP / LIME)**: Provide applicant-specific feature importance explanations showing exactly how credit history, income, and debt ratios affected the prediction.
- **Automated CI/CD**: Containerize with Docker and implement automated deployment pipelines via GitHub Actions and cloud services.

