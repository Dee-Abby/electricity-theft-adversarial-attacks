"""
SGCC preprocessing step 5: adversarial evasion attacks against the trained RF detector.

Threat model: attacker controls raw daily meter readings for their own account
(the real-world attack surface), NOT the engineered features directly. Each attack
perturbs raw readings -> re-runs feature engineering -> re-queries the RF.

Three attacker knowledge levels, following Takiddin et al.'s white/gray/black-box
taxonomy, adapted for a non-differentiable Random Forest:

- WHITE-BOX: attacker knows which features matter most (from feature_importances_)
  and targets peak/volatility-driving days directly.
- GRAY-BOX: attacker knows the detector uses consumption statistics (mean/std/peak-type
  features) but not exact importances - applies generic domain-realistic smoothing
  (NNP-style: each day replaced by local neighbor average).
- BLACK-BOX: attacker has query access only (prediction probability), no knowledge
  of features or importances - uses randomized greedy search, keeping perturbations
  only if they reduce the theft-probability score.

Requires: df_clean (cleaned raw daily data, indexed by CONS_NO), the trained RF,
and the engineer_features function from feature_engineering.py
"""

import numpy as np
import pandas as pd
from src.feature_engineering import engineer_features


def get_raw_readings(df_clean, customer_ids, label_col="FLAG"):
    """Slice out raw daily readings for a set of customers."""
    date_cols = [c for c in df_clean.columns if c != label_col]
    return df_clean.loc[customer_ids, date_cols].copy(), date_cols


# ---------------------------------------------------------------------------
# GRAY-BOX: domain-realistic neighbor smoothing (NNP-style)
# ---------------------------------------------------------------------------
def nnp_attack(raw_readings, window=3):
    """
    Nearest-neighbor perturbation: replace each day's reading with the mean of
    its +/- window neighboring days. Smooths peaks and reduces volatility using
    only the customer's own historical pattern - no model knowledge required
    beyond knowing the detector likely reacts to volatility/peaks (general domain
    knowledge, not exact feature weights).
    """
    arr = raw_readings.values.astype(float)
    smoothed = np.copy(arr)
    n_days = arr.shape[1]
    for j in range(n_days):
        lo, hi = max(0, j - window), min(n_days, j + window + 1)
        smoothed[:, j] = arr[:, lo:hi].mean(axis=1)
    return pd.DataFrame(smoothed, index=raw_readings.index, columns=raw_readings.columns)


# ---------------------------------------------------------------------------
# WHITE-BOX: targeted peak dampening using known feature importances
# ---------------------------------------------------------------------------
def white_box_attack(raw_readings, top_pct=0.15, dampen_to_percentile=50):
    """
    Attacker knows std/max/peak_to_mean dominate the detector's decision
    (from feature_importances_). Directly dampens the top_pct highest-consumption
    days toward the customer's own median, specifically targeting the features
    known to matter most rather than smoothing uniformly.
    """
    arr = raw_readings.values.astype(float)
    perturbed = np.copy(arr)
    n_customers, n_days = arr.shape
    n_top = max(1, int(n_days * top_pct))

    for i in range(n_customers):
        row = arr[i]
        median_val = np.median(row)
        top_idx = np.argsort(row)[-n_top:]
        perturbed[i, top_idx] = median_val + (row[top_idx] - median_val) * (1 - dampen_to_percentile / 100)

    return pd.DataFrame(perturbed, index=raw_readings.index, columns=raw_readings.columns)


# ---------------------------------------------------------------------------
# BLACK-BOX: query-based randomized greedy search
# ---------------------------------------------------------------------------
def black_box_attack(raw_readings, rf, batch_size=25, max_rounds=15, step_frac=0.1,
                      patience=5, random_state=42, label_col="FLAG"):
    """
    Attacker has query access to the detector's theft-probability output only.
    No knowledge of features or model internals.

    Batched randomized greedy search: at each round, generate `batch_size` candidate
    perturbations of the current best solution PER CUSTOMER, score the whole batch in
    ONE vectorized engineer_features + predict_proba call (not one-at-a-time), and keep
    whichever candidate reduced theft-probability most. This is the same search strategy
    as a per-query loop, just batched for speed - a single-row-at-a-time version of this
    took 45+ minutes on ~330 customers; this version does the same total work through
    vectorized pandas/sklearn calls instead of ~49,500 individual Python-level calls.

    max_rounds x batch_size = total candidates evaluated per customer (e.g. 15 x 25 = 375,
    comparable search budget to max_queries=150-375 in the old version, in a fraction of the time).
    """
    rng = np.random.default_rng(random_state)
    n_customers, n_days = raw_readings.shape
    current = raw_readings.values.astype(float).copy()

    def score_batch(arr_2d):
        """arr_2d: (n_rows, n_days) -> theft probabilities, via ONE batched pipeline call."""
        temp_df = pd.DataFrame(arr_2d, columns=raw_readings.columns)
        temp_df[label_col] = 0  # placeholder
        feats = engineer_features(temp_df, label_col=label_col, verbose=False)
        X = feats.drop(columns=[label_col])
        return rf.predict_proba(X)[:, 1]

    best_probs = score_batch(current)
    no_improve = np.zeros(n_customers, dtype=int)
    active = np.ones(n_customers, dtype=bool)

    for round_i in range(max_rounds):
        if not active.any():
            break
        active_idx = np.where(active)[0]

        # generate batch_size candidates for each active customer, evaluate all at once
        all_candidates = []
        owner_idx = []
        for ci in active_idx:
            base = current[ci]
            for _ in range(batch_size):
                cand = base.copy()
                n_touch = max(1, int(n_days * step_frac))
                idx = rng.choice(n_days, size=n_touch, replace=False)
                scale = rng.uniform(0.7, 0.98, size=n_touch)
                cand[idx] = cand[idx] * scale
                all_candidates.append(cand)
                owner_idx.append(ci)

        batch_arr = np.vstack(all_candidates)
        batch_probs = score_batch(batch_arr)

        owner_idx = np.array(owner_idx)
        for ci in active_idx:
            mask = owner_idx == ci
            cand_probs = batch_probs[mask]
            cand_rows = batch_arr[mask]
            best_in_batch = cand_probs.argmin()
            if cand_probs[best_in_batch] < best_probs[ci]:
                current[ci] = cand_rows[best_in_batch]
                best_probs[ci] = cand_probs[best_in_batch]
                no_improve[ci] = 0
            else:
                no_improve[ci] += 1
                if no_improve[ci] >= patience:
                    active[ci] = False

        print(f"Round {round_i+1}/{max_rounds}: {active.sum()} customers still improving, "
              f"mean theft-prob so far = {best_probs.mean():.4f}")

    return pd.DataFrame(current, index=raw_readings.index, columns=raw_readings.columns)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
def evaluate_attack(perturbed_raw, original_labels, rf, label_col="FLAG"):
    """
    Re-engineers features from perturbed raw readings, re-predicts with the RF,
    and reports how many theft customers successfully evaded detection.
    """
    temp = perturbed_raw.copy()
    temp[label_col] = original_labels.values
    feats = engineer_features(temp, label_col=label_col, verbose=False)
    X_new = feats.drop(columns=[label_col])

    y_pred_after = rf.predict(X_new)
    y_proba_after = rf.predict_proba(X_new)[:, 1]

    n_theft = (original_labels == 1).sum()
    n_evaded = ((original_labels == 1) & (y_pred_after == 0)).sum()
    evasion_rate = n_evaded / n_theft if n_theft > 0 else 0.0

    print(f"Theft customers attacked: {n_theft}")
    print(f"Successfully evaded detection: {n_evaded} ({evasion_rate*100:.1f}%)")
    print(f"Mean theft-probability after attack: {y_proba_after[original_labels.values == 1].mean():.4f}")

    return {
        "evasion_rate": evasion_rate,
        "n_theft": n_theft,
        "n_evaded": n_evaded,
        "y_pred_after": y_pred_after,
        "y_proba_after": y_proba_after,
    }


def make_query_fn(rf, label_col_placeholder="FLAG"):
    """Builds a query function usable by black_box_attack: raw 1-row df -> theft probability."""
    def query_fn(raw_row_df):
        temp = raw_row_df.copy()
        temp[label_col_placeholder] = 0  # placeholder, dropped before prediction
        feats = engineer_features(temp, label_col=label_col_placeholder, verbose=False)
        X_row = feats.drop(columns=[label_col_placeholder])
        return rf.predict_proba(X_row)[0, 1]
    return query_fn


def make_fast_query_fn(rf, date_cols, feature_order, weekday_mask):
    """
    Numpy-only query function - computes the same 12 features as engineer_features
    but skips all pandas DataFrame construction/copy overhead per call. Use this
    instead of make_query_fn() for black_box_attack when running on real data
    (hundreds of customers x hundreds of queries adds up fast with the pandas version).

    feature_order: list of feature column names in the exact order the RF was trained on
                   (e.g. list(X_train.columns))
    weekday_mask: boolean numpy array, same length as date_cols, True for weekday columns
                  (precompute once: pd.to_datetime(date_cols, format="%Y/%m/%d").dayofweek < 5)
    """
    from scipy.stats import skew as _skew

    def compute_features(row):
        # row: 1D numpy array of raw daily readings
        mean = row.mean()
        std = row.std(ddof=1)
        median = np.median(row)
        mx = row.max()
        mn = row.min()
        peak_to_mean = mx / mean if mean != 0 else 0.0
        sk = _skew(row)
        pct_zero = (row == 0).mean()
        weekday_mean = row[weekday_mask].mean()
        weekend_mean = row[~weekday_mask].mean()
        wd_we_ratio = weekday_mean / weekend_mean if weekend_mean != 0 else 0.0

        # rolling 7-day std, mean of it - TRAILING window (past 7 incl. current),
        # matching pandas .rolling(window=7, min_periods=3).std() (ddof=1) exactly
        n = len(row)
        window = 7
        rolling_stds = []
        for j in range(n):
            lo, hi = max(0, j - window + 1), j + 1
            seg = row[lo:hi]
            if len(seg) >= 3:
                rolling_stds.append(seg.std(ddof=1))
        mean_rolling7_std = np.mean(rolling_stds) if rolling_stds else 0.0

        feat_map = {
            "mean": mean, "std": std, "median": median, "max": mx, "min": mn,
            "peak_to_mean": peak_to_mean, "skewness": sk, "pct_zero_days": pct_zero,
            "weekday_mean": weekday_mean, "weekend_mean": weekend_mean,
            "weekday_weekend_ratio": wd_we_ratio, "mean_rolling7_std": mean_rolling7_std,
        }
        return np.array([feat_map[f] for f in feature_order])

    def query_fn(raw_row_df):
        row = raw_row_df.values[0].astype(float)
        feat_vec = compute_features(row).reshape(1, -1)
        feat_df = pd.DataFrame(feat_vec, columns=feature_order)
        return rf.predict_proba(feat_df)[0, 1]

    return query_fn


if __name__ == "__main__":
    print("Import attack functions into your notebook: nnp_attack, white_box_attack, "
          "black_box_attack, evaluate_attack, make_query_fn, get_raw_readings")