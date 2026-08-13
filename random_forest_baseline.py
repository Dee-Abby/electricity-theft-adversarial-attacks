"""
SGCC preprocessing step 4: stratified train/test split + Random Forest baseline.
Input: df_features (output of engineer_features) - one row per customer, FLAG + 12 features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score
)


def split_data(df_features, label_col="FLAG", test_size=0.2, random_state=42):
    X = df_features.drop(columns=[label_col])
    y = df_features[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"Train: {X_train.shape[0]} customers | Theft %: {y_train.mean()*100:.2f}")
    print(f"Test:  {X_test.shape[0]} customers | Theft %: {y_test.mean()*100:.2f}")

    return X_train, X_test, y_train, y_test


def train_rf_baseline(X_train, y_train, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_rf(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("RANDOM FOREST BASELINE - TEST SET PERFORMANCE")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Theft"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(f"                Pred Legit   Pred Theft")
    print(f"Actual Legit    {cm[0][0]:<12} {cm[0][1]}")
    print(f"Actual Theft    {cm[1][0]:<12} {cm[1][1]}")

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"\nROC-AUC: {auc:.4f}")
    print(f"F1 (theft class): {f1:.4f}")

    # feature importance, sorted
    importances = pd.Series(rf.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    return {"y_pred": y_pred, "y_proba": y_proba, "auc": auc, "f1": f1, "importances": importances}


def run_baseline_pipeline(df_features, label_col="FLAG"):
    X_train, X_test, y_train, y_test = split_data(df_features, label_col)
    rf = train_rf_baseline(X_train, y_train)
    results = evaluate_rf(rf, X_test, y_test)
    return rf, X_train, X_test, y_train, y_test, results


if __name__ == "__main__":
    print("Import run_baseline_pipeline(df_features) into your notebook.")