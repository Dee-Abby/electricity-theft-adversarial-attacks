"""
Statistical significance testing for evasion attack results.

Answers: is the gap between two attacks' evasion rates (e.g. white-box 74.5%
vs black-box 71.7%) a real effect, or could it be noise from this particular
train/test split and set of theft customers?

Requires per-customer attack outcomes - the y_pred_after array already
returned by evaluate_attack() in attack_evaluation.py for each attack.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def paired_evasion_arrays(y_test, theft_test_ids, y_pred_after_a, y_pred_after_b):
    """
    Builds two aligned boolean arrays: did attack A evade detection for this
    customer? did attack B? Both arrays are in the same customer order, so
    they can be directly paired.

    y_pred_after_a / b: the y_pred_after arrays returned by evaluate_attack()
                         for attack A and attack B respectively (predictions
                         on the SAME theft_test_ids, same order).
    """
    # y_pred_after is 0/1 for ALL customers passed into evaluate_attack;
    # "evaded" means predicted 0 (legitimate) when true label was 1 (theft)
    evaded_a = (y_pred_after_a == 0)
    evaded_b = (y_pred_after_b == 0)
    return evaded_a, evaded_b


def mcnemar_test(evaded_a, evaded_b, attack_a_name="Attack A", attack_b_name="Attack B"):
    """
    McNemar's test on paired binary outcomes (same customers, two attacks).
    Tests whether the DISAGREEMENTS between the two attacks are asymmetric
    (i.e. one attack is genuinely stronger) rather than just noise.
    """
    # contingency table: [[both evaded, A evaded only], [B evaded only, neither evaded]]
    both = np.sum(evaded_a & evaded_b)
    a_only = np.sum(evaded_a & ~evaded_b)
    b_only = np.sum(~evaded_a & evaded_b)
    neither = np.sum(~evaded_a & ~evaded_b)

    table = [[both, a_only], [b_only, neither]]
    result = mcnemar(table, exact=(min(a_only, b_only) < 25))

    print(f"McNemar's test: {attack_a_name} vs {attack_b_name}")
    print(f"  Both evaded: {both} | {attack_a_name} only: {a_only} | "
          f"{attack_b_name} only: {b_only} | Neither: {neither}")
    print(f"  Statistic: {result.statistic:.4f}  p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print(f"  -> Significant difference (p < 0.05): the gap between these attacks is unlikely to be noise.")
    else:
        print(f"  -> NOT significant (p >= 0.05): cannot rule out that the gap is due to chance.")
    print()
    return result


def bootstrap_evasion_ci(evaded, n_bootstrap=2000, ci=95, random_state=42, label="Attack"):
    """
    Bootstrap confidence interval for a single attack's evasion rate.
    Resamples the customer set with replacement n_bootstrap times, recomputes
    the evasion rate each time, and reports the percentile CI.
    """
    rng = np.random.default_rng(random_state)
    n = len(evaded)
    boot_rates = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        boot_rates[i] = evaded[sample_idx].mean()

    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    lower, upper = np.percentile(boot_rates, [lower_pct, upper_pct])
    point_estimate = evaded.mean()

    print(f"{label}: evasion rate = {point_estimate*100:.1f}%  "
          f"[{ci}% CI: {lower*100:.1f}% - {upper*100:.1f}%]")

    return point_estimate, lower, upper


def compare_all_attacks(y_test, theft_test_ids, results_dict):
    """
    Full significance comparison across all three attacks.

    results_dict: {"White-box": wb_results, "Gray-box": nnp_results, "Black-box": bb_results}
                  each results dict must contain "y_pred_after" for the SAME
                  theft_test_ids in the SAME order.
    """
    print("=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)
    evaded = {}
    for name, res in results_dict.items():
        e = (res["y_pred_after"] == 0)
        evaded[name] = e
        bootstrap_evasion_ci(e, label=name)

    print("\n" + "=" * 60)
    print("PAIRWISE MCNEMAR'S TESTS")
    print("=" * 60)
    names = list(results_dict.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mcnemar_test(evaded[names[i]], evaded[names[j]], names[i], names[j])


if __name__ == "__main__":
    print("Import compare_all_attacks(y_test, theft_test_ids, results_dict) into your notebook.")