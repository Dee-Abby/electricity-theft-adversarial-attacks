"""
SGCC preprocessing step 4b: quick hyperparameter tuning for the RF baseline.
Uses RandomizedSearchCV (fast) rather than exhaustive GridSearchCV, scored on
F1 for the theft (minority) class specifically - not accuracy, which is
misleading on imbalanced data.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, fbeta_score

# scorer targeting theft class (label=1) specifically
theft_f1_scorer = make_scorer(f1_score, pos_label=1)

# F2 weighs recall twice as heavily as precision - matches the operational
# priority already established (catching theft > avoiding false positives)
theft_f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

PARAM_DIST = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [None, 10, 15, 20, 30],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": ["balanced", "balanced_subsample", {0: 1, 1: 3}, {0: 1, 1: 5}],
}

# Round 2: wider space (more estimators/depth options, added bootstrap and
# criterion) and more iterations - a genuinely different search, not a repeat
PARAM_DIST_ROUND2 = {
    "n_estimators": [200, 300, 500, 800, 1000, 1200],
    "max_depth": [None, 8, 10, 15, 20, 30, 40],
    "min_samples_leaf": [1, 2, 3, 5, 8, 10, 15],
    "min_samples_split": [2, 5, 10, 15],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight": ["balanced", "balanced_subsample", {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 8}],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
}


def tune_rf(X_train, y_train, n_iter=25, random_state=42):
    base_rf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_rf,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring=theft_f1_scorer,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best theft-F1 (CV): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_rf_round2_f2(X_train, y_train, n_iter=50, random_state=42):
    """
    Second tuning pass: wider hyperparameter space, more iterations, and
    scored on F2 (recall-weighted) instead of F1 - directly optimizes for
    catching more theft, consistent with the recall-priority decision
    already made when selecting the round-1 tuned model.
    """
    base_rf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_rf,
        param_distributions=PARAM_DIST_ROUND2,
        n_iter=n_iter,
        scoring=theft_f2_scorer,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best theft-F2 (CV): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


if __name__ == "__main__":
    print("Import tune_rf(X_train, y_train) into your notebook.")