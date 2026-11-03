"""
Script to generate the complete Jupyter Notebook ml/notebooks/loan_approval_analysis.ipynb.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Loan Approval Prediction Using K-Nearest Neighbors (KNN)
## Machine Learning Capstone Project

**Author:** Senior Machine Learning Engineer & Data Scientist  
**Dataset:** Kaggle Loan Prediction Dataset (`train.csv`)  
**Objective:** Build an end-to-end classification system predicting loan approvals using K-Nearest Neighbors (KNN), featuring rigorous preprocessing, hyperparameter tuning, financial risk evaluation, and model serialization.
"""))

# Section 1
cells.append(nbf.v4.new_markdown_cell("""## Phase 1: Dataset Acquisition and Exploratory Data Analysis (EDA)
In this phase, we load the raw `train.csv` dataset, inspect its structure, data types, missing values, duplicate entries, and target variable distribution.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.titlesize": 14})

# 1.1 Load dataset
df = pd.read_csv('../data/train.csv')
print(f"Dataset Shape: {df.shape}")
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# 1.2 Column names, data types, and non-null counts
df.info()
"""))

cells.append(nbf.v4.new_code_cell("""# 1.3 Statistical summary of numerical and categorical columns
df.describe(include='all')
"""))

cells.append(nbf.v4.new_code_cell("""# 1.4 Missing values analysis and duplicate checks
print("=== Missing Values ===")
print(df.isnull().sum())
print(f"\\nDuplicate Records: {df.duplicated().sum()}")
"""))

cells.append(nbf.v4.new_code_cell("""# 1.5 Target variable (Loan_Status) distribution
approval_counts = df['Loan_Status'].value_counts()
approval_pcts = df['Loan_Status'].value_counts(normalize=True) * 100
print("=== Target Distribution ===")
for status, count in approval_counts.items():
    print(f"Status '{status}': {count} applicants ({approval_pcts[status]:.2f}%)")
"""))

# Section 2
cells.append(nbf.v4.new_markdown_cell("""## Phase 2: Data Preprocessing Pipeline
To prevent data leakage, all preprocessing transformations are encapsulated within a `ColumnTransformer`:
- Numerical features: Median Imputation + `StandardScaler`
- Categorical features: Most Frequent Imputation + `OneHotEncoder(handle_unknown='ignore')`
- Drop identifier column `Loan_ID`
- Stratified 80/20 train-test split with `random_state=42`
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define feature sets
CATEGORICAL_FEATURES = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERICAL_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Separate features (X) and target (y)
X = df[ALL_FEATURES].copy()
y = df['Loan_Status'].map({'N': 0, 'Y': 1}).astype(int)

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set:  X_test={X_test.shape}, y_test={y_test.shape}")
"""))

cells.append(nbf.v4.new_code_cell("""# Build preprocessing ColumnTransformer
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, NUMERICAL_FEATURES),
    ('cat', cat_pipeline, CATEGORICAL_FEATURES)
])
"""))

# Section 3
cells.append(nbf.v4.new_markdown_cell("""## Phase 3: Train Default K-Nearest Neighbors (KNN) Model & Compare Specific K Values
We train the baseline KNN classifier (default `n_neighbors=5`) inside the pipeline and evaluate on the untouched test set.
Then we compare K = 3, 5, and 7.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

default_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

default_pipeline.fit(X_train, y_train)
y_pred_default = default_pipeline.predict(X_test)
acc_default = accuracy_score(y_test, y_pred_default)

print(f"Default KNN (K=5) Test Accuracy: {acc_default * 100:.2f}%\\n")
print("=== Default Model Classification Report ===")
print(classification_report(y_test, y_pred_default, target_names=['Rejected', 'Approved']))
"""))

cells.append(nbf.v4.new_code_cell("""# Experiment with K = 3, 5, 7
k_experiments = {}
for k in [3, 5, 7]:
    k_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])
    k_pipe.fit(X_train, y_train)
    k_acc = k_pipe.score(X_test, y_test)
    k_experiments[k] = k_acc

print("K-Value Comparison Table:")
print("| K Value | Accuracy |")
print("|---------|----------|")
for k, acc in k_experiments.items():
    print(f"| {k:<7} | {acc * 100:.2f}%   |")
"""))

# Section 4
cells.append(nbf.v4.new_markdown_cell("""## Phase 4: Hyperparameter Tuning via GridSearchCV
We run a 5-fold Stratified Cross-Validation search across K values from 1 to 20, distance metrics (Euclidean, Manhattan), and weighting schemes (Uniform, Distance).
Importantly, preprocessing is inside the pipeline, guaranteeing proper fold isolation without data leakage.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'classifier__n_neighbors': list(range(1, 21)),
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=default_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best 5-Fold CV Accuracy: {grid_search.best_score_ * 100:.2f}%")
"""))

# Section 5
cells.append(nbf.v4.new_markdown_cell("""## Phase 5: Model Evaluation & Financial Risk Analysis
Evaluating the optimized model on the untouched test set (123 applicants). In loan approval, False Positives (predicting Approved when the applicant will default) represent severe financial credit risk.
"""))

cells.append(nbf.v4.new_code_cell("""best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)

cm = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print(f"Optimized Model Test Accuracy: {acc_tuned * 100:.2f}%")
print("\\n=== Optimized Model Classification Report ===")
print(classification_report(y_test, y_pred_tuned, target_names=['Rejected', 'Approved']))

print("\\n=== Financial Risk Breakdown ===")
print(f"True Positives (Correct Approvals):       {tp}")
print(f"True Negatives (Correct Rejections):       {tn}")
print(f"False Approvals (High Default Risk / FP): {fp}")
print(f"False Rejections (Lost Revenue / FN):     {fn}")
"""))

# Section 6
cells.append(nbf.v4.new_markdown_cell("""## Phase 6: Visualizations
Displaying the generated figures saved in `ml/visualizations/`.
"""))

cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import glob

vis_files = [
    '../visualizations/loan_approval_distribution.png',
    '../visualizations/missing_values_visualization.png',
    '../visualizations/numerical_feature_distributions.png',
    '../visualizations/correlation_heatmap.png',
    '../visualizations/k_value_accuracy_comparison.png',
    '../visualizations/confusion_matrix_default_knn.png',
    '../visualizations/confusion_matrix_optimized_knn.png',
]

for img_path in vis_files:
    print(f"Visualizing: {img_path}")
    display(Image(filename=img_path))
"""))

# Section 7
cells.append(nbf.v4.new_markdown_cell("""## Phase 7: Model Serialization
Serialize the best pipeline into `best_knn_model.joblib` and save metadata to `model_config.json`.
"""))

cells.append(nbf.v4.new_code_cell("""import joblib

joblib.dump(best_model, '../models/best_knn_model.joblib')
print("Model pipeline successfully serialized to ml/models/best_knn_model.joblib")
"""))

nb["cells"] = cells

with open("ml/notebooks/loan_approval_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Created ml/notebooks/loan_approval_analysis.ipynb")

