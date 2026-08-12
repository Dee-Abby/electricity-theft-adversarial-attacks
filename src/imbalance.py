"""
Class imbalance comparison: random undersampling vs. class-weighting.

Addresses the methodological gap: class_weight='balanced' was the only
imbalance-handling strategy tested. This adds random undersampling (RUS) of
the majority (legitimate) class as a second strategy, trained and evaluated
on the SAME train/test split as the existing models, for a fair comparison.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from src.random_forest_baseline import evaluate_rf


def random_undersample(X_train, y_train, random_state=42):
    """
    Randomly drops majority-class (legitimate) rows until both classes are
    equal in count. Minority (theft) class is left untouched - every real
    theft example in training is kept.
    """
    rng = np.random.default_rng(random_state)

    theft_idx = y_train[y_train == 1].index
    legit_idx = y_train[y_train == 0].index

    n_theft = len(theft_idx)
    sampled_legit_idx = rng.choice(legit_idx, size=n_theft, replace=False)

    keep_idx = np.concatenate([theft_idx.values, sampled_legit_idx])
    rng.shuffle(keep_idx)

    X_bal = X_train.loc[keep_idx]
    y_bal = y_train.loc[keep_idx]

    print(f"Undersampled training set: {len(X_bal)} rows "
          f"({(y_bal == 1).sum()} theft, {(y_bal == 0).sum()} legitimate — now 50/50)")

    return X_bal, y_bal


def smote_oversample(X_train, y_train, random_state=42, k_neighbors=5):
    """
    SMOTE (Synthetic Minority Oversampling Technique): generates NEW synthetic
    theft examples by interpolating between a real theft customer's feature
    vector and one of its k-nearest theft neighbors in feature space - not
    simple duplication. Majority (legitimate) class is left untouched;
    minority (theft) class is synthetically expanded to match its count.
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    n_synthetic = (y_bal == 1).sum() - (y_train == 1).sum()
    print(f"SMOTE training set: {len(X_bal)} rows "
          f"({(y_bal == 1).sum()} theft [{n_synthetic} synthetic], "
          f"{(y_bal == 0).sum()} legitimate — now 50/50)")

    return X_bal, y_bal


def train_rf_undersampled(X_train, y_train, random_state=42, **rf_kwargs):
    """
    Trains an RF on a randomly undersampled (balanced) training set.
    No class_weight needed here - the balancing already happened in the data,
    not via loss re-weighting, so this is a genuinely different mechanism
    than the class_weight='balanced' models.
    """
    X_bal, y_bal = random_undersample(X_train, y_train, random_state)

    defaults = dict(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=random_state)
    defaults.update(rf_kwargs)

    rf = RandomForestClassifier(**defaults)
    rf.fit(X_bal, y_bal)
    return rf


def train_rf_smote(X_train, y_train, random_state=42, **rf_kwargs):
    """
    Trains an RF on a SMOTE-oversampled (balanced) training set. Like
    undersampling, no class_weight needed - balancing happens in the data.
    Unlike undersampling, ALL original legitimate rows are kept; only
    synthetic theft rows are added.
    """
    X_bal, y_bal = smote_oversample(X_train, y_train, random_state)

    defaults = dict(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=random_state)
    defaults.update(rf_kwargs)

    rf = RandomForestClassifier(**defaults)
    rf.fit(X_bal, y_bal)
    return rf


def compare_imbalance_strategies(X_train, y_train, X_test, y_test,
                                  weighted_rf=None, random_state=42):
    """
    Runs (or reuses) a class-weighted RF, a randomly undersampled RF, and a
    SMOTE-oversampled RF, all evaluated on the identical test set, and prints
    all three reports side by side for direct comparison.

    weighted_rf: pass your existing tuned/baseline RF (already trained with
                 class_weight) to reuse it instead of retraining.
    """
    if weighted_rf is None:
        print("No existing weighted model passed - training a fresh class_weight='balanced' RF...")
        weighted_rf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=random_state
        )
        weighted_rf.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("STRATEGY 1: class_weight='balanced' (existing approach)")
    print("=" * 60)
    results_weighted = evaluate_rf(weighted_rf, X_test, y_test)

    print("\n" + "=" * 60)
    print("STRATEGY 2: Random undersampling")
    print("=" * 60)
    undersampled_rf = train_rf_undersampled(X_train, y_train, random_state=random_state)
    results_undersampled = evaluate_rf(undersampled_rf, X_test, y_test)

    print("\n" + "=" * 60)
    print("STRATEGY 3: SMOTE oversampling")
    print("=" * 60)
    smote_rf = train_rf_smote(X_train, y_train, random_state=random_state)
    results_smote = evaluate_rf(smote_rf, X_test, y_test)

    return {
        "weighted_rf": weighted_rf, "results_weighted": results_weighted,
        "undersampled_rf": undersampled_rf, "results_undersampled": results_undersampled,
        "smote_rf": smote_rf, "results_smote": results_smote,
    }


if __name__ == "__main__":
    print("Import compare_imbalance_strategies(X_train, y_train, X_test, y_test, weighted_rf=best_rf) "
          "into your notebook.")