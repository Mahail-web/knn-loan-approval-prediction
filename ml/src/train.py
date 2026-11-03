"""
Model Training & Hyperparameter Tuning module using K-Nearest Neighbors (KNN).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, split_data, prepare_features_and_target, load_data


def train_default_knn(X_train, y_train):
    """
    Train default KNN classifier pipeline.
    Default parameters: n_neighbors=5, metric='minkowski', p=2, weights='uniform'.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", KNeighborsClassifier()),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_specific_k_values(X_train, y_train, X_test, y_test, k_values=[3, 5, 7]):
    """
    Train separate KNN pipelines for specified K values and record test accuracies.
    """
    results = {}
    models = {}
    for k in k_values:
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        results[k] = float(acc)
        models[k] = pipe
    return results, models


def tune_knn_hyperparameters(X_train, y_train, k_range=range(1, 21), cv_folds=5):
    """
    Optimize n_neighbors hyperparameter using GridSearchCV with StratifiedKFold.
    Preprocessors are fitted strictly within each cross-validation fold.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "classifier__n_neighbors": list(k_range),
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search

