"""
SGCC preprocessing step 2: date reordering, sparse-customer drop, interpolation.
Assumes df_elect is indexed by CONS_NO with FLAG + date-string columns.
"""

import pandas as pd
import numpy as np


def fix_date_order(df, label_col="FLAG"):
    """Sort date columns chronologically instead of string/alphabetical order."""
    date_cols = [c for c in df.columns if c != label_col]
    parsed = pd.to_datetime(date_cols, format="%Y/%m/%d")
    sorted_cols = [c for _, c in sorted(zip(parsed, date_cols))]
    return df[[label_col] + sorted_cols]


def drop_sparse_customers(df, label_col="FLAG", threshold_pct=80):
    """Drop customers missing more than threshold_pct of their readings."""
    date_cols = [c for c in df.columns if c != label_col]
    missing_pct = df[date_cols].isna().mean(axis=1) * 100
    kept = df[missing_pct <= threshold_pct].copy()
    print(f"Dropped {len(df) - len(kept)} customers (>{threshold_pct}% missing). "
          f"Remaining: {len(kept)}")
    return kept


def interpolate_readings(df, label_col="FLAG", limit=7):
    """
    Linear interpolation along the (now correctly ordered) date axis, per customer.
    limit=7 caps how many consecutive missing days get filled, so we don't
    fabricate long stretches of fake data.
    Remaining edge NaNs (leading/trailing) are filled with 0 as last resort.
    """
    date_cols = [c for c in df.columns if c != label_col]
    interpolated = df[date_cols].interpolate(
        method="linear", axis=1, limit=limit, limit_direction="both"
    )
    interpolated = interpolated.fillna(0)
    result = df[[label_col]].join(interpolated)
    remaining_na = result[date_cols].isna().sum().sum()
    print(f"Remaining NaNs after interpolation + fill: {remaining_na}")
    return result


def run_full_cleaning(df, label_col="FLAG", drop_threshold=80, interp_limit=7):
    print("Step 1: fixing date order...")
    df = fix_date_order(df, label_col)

    print("Step 2: dropping sparse customers...")
    df = drop_sparse_customers(df, label_col, drop_threshold)

    print("Step 3: interpolating remaining gaps...")
    df = interpolate_readings(df, label_col, interp_limit)

    theft_pct = (df[label_col].mean() * 100)
    print(f"Final shape: {df.shape}  |  Theft share: {theft_pct:.2f}%")
    return df


if __name__ == "__main__":
    print("Import run_full_cleaning(df) into your notebook.")