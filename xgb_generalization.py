"""
Extension: test whether the evasion pattern found on Random Forest also
holds on a second, structurally different tree ensemble (XGBoost).

Same 12 features, same train/test split, same three attack mechanisms -
only the detector changes. Gray-box and black-box attacks need NO code
changes (they only call predict_proba). White-box needs the new model's
own feature_importances_ instead of RF's.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from random_forest_baseline import evaluate_rf
from attack_evaluation import (
    get_raw_readings, nnp_attack, white_box_attack, black_box_attack, evaluate_attack
)


def train_xgb_baseline(X_train, y_train, random_state=42, **xgb_kwargs):
    """
    XGBoost detector using the same 12 features as the RF study.
    scale_pos_weight is XGBoost's equivalent of class_weight='balanced':
    it upweights the minority (theft) class in the loss function.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    defaults = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=random_state, n_jobs=-1,
    )
    defaults.update(xgb_kwargs)

    xgb = XGBClassifier(**defaults)
    xgb.fit(X_train, y_train)
    return xgb


def white_box_attack_xgb(raw_readings, xgb_model, feature_order, top_pct=0.15, dampen_to_percentile=50):
    """
    Same mechanism as the RF white-box attack, but reads feature importance
    from the XGBoost model instead of assuming RF's specific ranking.
    """
    importances = pd.Series(xgb_model.feature_importances_, index=feature_order)
    top_features = importances.sort_values(ascending=False).head(3).index.tolist()
    print(f"XGBoost's top 3 features (used to inform targeted attack): {top_features}")

    # the attack mechanism itself (dampen highest-consumption days) is
    # detector-agnostic - it doesn't need per-feature targeting to work,
    # since std/max/peak-type features are all driven by the same raw peaks
    return white_box_attack(raw_readings, top_pct=top_pct, dampen_to_percentile=dampen_to_percentile)


def run_xgb_generalization_study(df_clean, X_train, y_train, X_test, y_test, feature_order):
    """
    Full pipeline: train XGBoost, evaluate baseline, run all three attacks,
    and print a side-by-side comparison against what you'd expect from the
    RF results (58.7% / 74.5% / 71.7%).
    """
    print("Training XGBoost detector...")
    xgb_model = train_xgb_baseline(X_train, y_train)

    print("\n" + "=" * 60)
    print("XGBOOST BASELINE PERFORMANCE")
    print("=" * 60)
    baseline_results = evaluate_rf(xgb_model, X_test, y_test)  # works generically, not RF-specific

    theft_test_ids = y_test[y_test == 1].index
    raw_readings, _ = get_raw_readings(df_clean, theft_test_ids)

    print("\n" + "=" * 60)
    print("GRAY-BOX: NNP-style neighbor smoothing (XGBoost)")
    print("=" * 60)
    nnp_out = nnp_attack(raw_readings, window=3)
    nnp_results = evaluate_attack(nnp_out, y_test.loc[theft_test_ids], xgb_model)

    print("\n" + "=" * 60)
    print("WHITE-BOX: targeted peak dampening (XGBoost)")
    print("=" * 60)
    wb_out = white_box_attack_xgb(raw_readings, xgb_model, feature_order, top_pct=0.15)
    wb_results = evaluate_attack(wb_out, y_test.loc[theft_test_ids], xgb_model)

    print("\n" + "=" * 60)
    print("BLACK-BOX: batched query search (XGBoost)")
    print("=" * 60)
    bb_out = black_box_attack(raw_readings, xgb_model, batch_size=25, max_rounds=15, patience=5)
    bb_results = evaluate_attack(bb_out, y_test.loc[theft_test_ids], xgb_model)

    print("\n" + "=" * 60)
    print("SUMMARY: XGBoost vs Random Forest evasion rates")
    print("=" * 60)
    print(f"{'Attack':<20}{'XGBoost':<12}{'Random Forest (reference)':<25}")
    print(f"{'Gray-box':<20}{nnp_results['evasion_rate']*100:<12.1f}{'58.7':<25}")
    print(f"{'White-box':<20}{wb_results['evasion_rate']*100:<12.1f}{'74.5':<25}")
    print(f"{'Black-box':<20}{bb_results['evasion_rate']*100:<12.1f}{'71.7':<25}")

    return {
        "xgb_model": xgb_model, "baseline_results": baseline_results,
        "nnp_results": nnp_results, "wb_results": wb_results, "bb_results": bb_results,
    }


if __name__ == "__main__":
    print("Import run_xgb_generalization_study(df_clean, X_train, y_train, X_test, y_test, feature_order) "
          "into your notebook.")