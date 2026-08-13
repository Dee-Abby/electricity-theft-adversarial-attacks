"""
SGCC preprocessing step 3: feature engineering.
Input: df_clean (output of run_full_cleaning) - indexed by CONS_NO, FLAG + chronological date columns.
Output: a flat feature table ready for train/test split + Random Forest.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew


def engineer_features(df, label_col="FLAG", verbose=True):
    date_cols = [c for c in df.columns if c != label_col]
    dates = pd.to_datetime(date_cols, format="%Y/%m/%d")
    X = df[date_cols]

    feats = pd.DataFrame(index=df.index)

    feats["mean"] = X.mean(axis=1)
    feats["std"] = X.std(axis=1)
    feats["median"] = X.median(axis=1)
    feats["max"] = X.max(axis=1)
    feats["min"] = X.min(axis=1)
    feats["peak_to_mean"] = feats["max"] / feats["mean"].replace(0, np.nan)
    feats["skewness"] = X.apply(lambda row: skew(row.values), axis=1)
    feats["pct_zero_days"] = (X == 0).mean(axis=1)

    # weekday vs weekend split
    weekday_mask = np.array(dates.dayofweek < 5)
    feats["weekday_mean"] = X.loc[:, weekday_mask].mean(axis=1)
    feats["weekend_mean"] = X.loc[:, ~weekday_mask].mean(axis=1)
    feats["weekday_weekend_ratio"] = (
        feats["weekday_mean"] / feats["weekend_mean"].replace(0, np.nan)
    )

    # rolling 7-day std, averaged -> short-term volatility signature
    rolling_std = X.T.rolling(window=7, min_periods=3).std().T
    feats["mean_rolling7_std"] = rolling_std.mean(axis=1)

    # clean up any inf/nan produced by ratio divisions (e.g. mean==0 customers)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(feats.median(numeric_only=True))

    feats[label_col] = df[label_col].values

    if verbose:
        print(f"Feature table shape: {feats.shape}")
        print(f"Features: {[c for c in feats.columns if c != label_col]}")
        print(f"NaNs remaining: {feats.isna().sum().sum()}")

    return feats


if __name__ == "__main__":
    print("Import engineer_features(df_clean) into your notebook.")