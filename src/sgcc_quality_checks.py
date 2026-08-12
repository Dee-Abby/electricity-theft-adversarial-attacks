"""
SGCC dataset quality checks (5-9).
Assumes: df is your loaded dataframe, customer ID as index, 'FLAG' column present,
remaining columns are date-string consumption readings.
"""

import pandas as pd
import numpy as np

def run_quality_checks(df, label_col="FLAG"):
    date_cols = [c for c in df.columns if c != label_col]
    report = {}

    # 5. Duplicate columns (watch for pandas auto-renaming like '2016/9/28.1')
    suspect_dupes = [c for c in date_cols if "." in c and c.split(".")[0] in date_cols]
    report["suspected_duplicate_columns"] = suspect_dupes
    report["n_suspected_duplicate_columns"] = len(suspect_dupes)

    # Also check for genuinely repeated column names (rare but possible on raw load)
    col_counts = pd.Series(df.columns).value_counts()
    report["truly_duplicate_column_names"] = col_counts[col_counts > 1].index.tolist()

    # 6. Data types - flag any consumption column that isn't numeric
    non_numeric_cols = [c for c in date_cols if not pd.api.types.is_numeric_dtype(df[c])]
    report["non_numeric_columns"] = non_numeric_cols
    report["n_non_numeric_columns"] = len(non_numeric_cols)

    # 7. Negative values
    numeric_date_cols = [c for c in date_cols if pd.api.types.is_numeric_dtype(df[c])]
    neg_counts = (df[numeric_date_cols] < 0).sum()
    cols_with_negatives = neg_counts[neg_counts > 0]
    report["columns_with_negative_values"] = cols_with_negatives.to_dict()
    report["total_negative_readings"] = int(neg_counts.sum())

    # 8. Per-customer (row-wise) missingness
    row_missing_pct = df[date_cols].isna().mean(axis=1) * 100
    report["row_missing_pct_summary"] = row_missing_pct.describe().to_dict()
    high_missing_customers = row_missing_pct[row_missing_pct > 50]
    report["n_customers_over_50pct_missing"] = len(high_missing_customers)
    report["n_customers_over_80pct_missing"] = int((row_missing_pct > 80).sum())

    # 9. Duplicate customer IDs (index)
    dup_ids = df.index[df.index.duplicated(keep=False)]
    report["n_duplicate_customer_ids"] = len(dup_ids.unique())
    if len(dup_ids) > 0:
        # check if duplicated IDs have conflicting FLAG values
        conflicting = []
        for cid in dup_ids.unique():
            flags = df.loc[cid, label_col]
            if isinstance(flags, pd.Series) and flags.nunique() > 1:
                conflicting.append(cid)
        report["duplicate_ids_with_conflicting_flags"] = conflicting

    return report


def check_missingness_by_class(df, label_col="FLAG"):
    """
    Checks whether missingness is skewed between legitimate (0) and theft (1) customers.
    This matters because an aggressive drop-threshold could disproportionately
    remove theft cases, biasing the dataset against detection.
    """
    date_cols = [c for c in df.columns if c != label_col]
    row_missing_pct = df[date_cols].isna().mean(axis=1) * 100

    result = {}
    for flag_val, label in [(0, "legitimate"), (1, "theft")]:
        subset = row_missing_pct[df[label_col] == flag_val]
        result[label] = {
            "n_customers": len(subset),
            "mean_missing_pct": subset.mean(),
            "median_missing_pct": subset.median(),
            "n_over_50pct_missing": int((subset > 50).sum()),
            "n_over_80pct_missing": int((subset > 80).sum()),
        }

    keep_mask = row_missing_pct <= 80
    before = df[label_col].value_counts(normalize=True) * 100
    after = df.loc[keep_mask, label_col].value_counts(normalize=True) * 100
    result["theft_pct_before_drop"] = before.get(1, 0)
    result["theft_pct_after_80pct_drop"] = after.get(1, 0)

    return result


def print_missingness_by_class(result):
    print("=" * 60)
    print("MISSINGNESS BY CLASS (legitimate vs theft)")
    print("=" * 60)
    for label in ["legitimate", "theft"]:
        r = result[label]
        print(f"\n{label.upper()} (n={r['n_customers']})")
        print(f"    Mean missing %: {r['mean_missing_pct']:.2f}  |  Median: {r['median_missing_pct']:.2f}")
        print(f"    >50% missing: {r['n_over_50pct_missing']}  |  >80% missing: {r['n_over_80pct_missing']}")
    print(f"\nTheft % of dataset BEFORE any drop: {result['theft_pct_before_drop']:.2f}%")
    print(f"Theft % of dataset AFTER dropping >80%-missing customers: {result['theft_pct_after_80pct_drop']:.2f}%")
    print("=" * 60)


def plot_missingness_by_class(result, save_path="missingness_by_class.png"):
    """
    Produces a two-panel figure suitable for a supervisor meeting:
    (left) mean/median missingness compared across legitimate vs theft customers
    (right) theft's share of the dataset before vs after an 80%-missing drop threshold
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: missingness comparison
    labels = ["Legitimate", "Theft"]
    means = [result["legitimate"]["mean_missing_pct"], result["theft"]["mean_missing_pct"]]
    medians = [result["legitimate"]["median_missing_pct"], result["theft"]["median_missing_pct"]]

    x = np.arange(len(labels))
    width = 0.35
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, means, width, label="Mean missing %", color="#4C72B0")
    bars2 = ax1.bar(x + width/2, medians, width, label="Median missing %", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Missing days (%)")
    ax1.set_title("Per-Customer Missingness by Class")
    ax1.legend()
    for bars in (bars1, bars2):
        for b in bars:
            ax1.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width()/2, b.get_height()),
                         ha="center", va="bottom", fontsize=9)

    # Panel 2: theft share before/after drop
    ax2 = axes[1]
    stages = ["Before drop", "After dropping\n>80% missing"]
    theft_pcts = [result["theft_pct_before_drop"], result["theft_pct_after_80pct_drop"]]
    bars3 = ax2.bar(stages, theft_pcts, color=["#55A868", "#C44E52"])
    ax2.set_ylabel("Theft share of dataset (%)")
    ax2.set_title("Class Balance Impact of Drop Threshold")
    for b in bars3:
        ax2.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")


def print_report(report):
    print("=" * 60)
    print("SGCC DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\n[5] Duplicate columns")
    print(f"    Suspected (pandas-renamed) duplicates: {report['n_suspected_duplicate_columns']}")
    if report["suspected_duplicate_columns"]:
        print(f"    Examples: {report['suspected_duplicate_columns'][:5]}")
    print(f"    Truly duplicate column names: {report['truly_duplicate_column_names']}")

    print(f"\n[6] Non-numeric consumption columns")
    print(f"    Count: {report['n_non_numeric_columns']}")
    if report["non_numeric_columns"]:
        print(f"    Examples: {report['non_numeric_columns'][:5]}")

    print(f"\n[7] Negative values")
    print(f"    Total negative readings across dataset: {report['total_negative_readings']}")
    print(f"    Columns affected: {len(report['columns_with_negative_values'])}")

    print(f"\n[8] Row-wise (per-customer) missingness")
    s = report["row_missing_pct_summary"]
    print(f"    Mean missing %: {s['mean']:.2f}  |  Max: {s['max']:.2f}  |  Median: {s['50%']:.2f}")
    print(f"    Customers with >50% missing days: {report['n_customers_over_50pct_missing']}")
    print(f"    Customers with >80% missing days: {report['n_customers_over_80pct_missing']}")

    print(f"\n[9] Duplicate customer IDs")
    print(f"    Duplicate IDs found: {report['n_duplicate_customer_ids']}")
    if report["n_duplicate_customer_ids"] > 0:
        print(f"    IDs with conflicting FLAG values: {report.get('duplicate_ids_with_conflicting_flags')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Import run_quality_checks(df) and print_report(report) into your notebook/script.")