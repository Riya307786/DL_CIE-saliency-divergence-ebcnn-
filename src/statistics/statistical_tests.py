import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any

def friedman_global_test(methods_results: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Computes Friedman test for global comparison across multiple paired methods over seeds.
    methods_results: {method_name: [metric_seed1, metric_seed2, ...]}
    """
    method_names = list(methods_results.keys())
    data = [methods_results[m] for m in method_names]
    
    stat, p_val = stats.friedmanchisquare(*data)
    
    # Calculate average ranks
    arr = np.array(data) # (k, n_seeds)
    # rank within each seed (column)
    ranks = np.zeros_like(arr)
    for col in range(arr.shape[1]):
        # Higher is better -> rank descending
        ranks[:, col] = stats.rankdata(-arr[:, col])
    mean_ranks = {method_names[i]: float(np.mean(ranks[i, :])) for i in range(len(method_names))}

    return {
        "test": "Friedman Test",
        "statistic": float(stat),
        "p_value": float(p_val),
        "is_significant_05": bool(p_val < 0.05),
        "mean_ranks": mean_ranks,
        "n_seeds": len(data[0]),
        "n_methods": len(method_names)
    }

def wilcoxon_pairwise_tests(
    methods_results: Dict[str, List[float]],
    baseline_method: str = "Original_EBCNN"
) -> Dict[str, Any]:
    """
    Computes Wilcoxon signed-rank tests comparing candidate methods against baseline,
    with Holm-Bonferroni correction and rank-biserial correlation effect size.
    """
    base_scores = np.array(methods_results[baseline_method])
    comparisons = []
    
    other_methods = [m for m in methods_results if m != baseline_method]
    
    for m in other_methods:
        cand_scores = np.array(methods_results[m])
        diff = cand_scores - base_scores
        n = len(diff)
        
        # Check if identical
        if np.all(diff == 0):
            stat, p_val = 0.0, 1.0
            r_rb = 0.0
        else:
            try:
                res = stats.wilcoxon(cand_scores, base_scores, alternative="two-sided")
                stat = float(res.statistic)
                p_val = float(res.pvalue)
            except ValueError:
                stat = 0.0
                p_val = 1.0
                
            # Rank-biserial correlation
            # r_rb = 4 * (sum of positive ranks - sum of negative ranks) / ...
            # or r_rb = 1 - 2*W / (n*(n+1)/2)
            total_rank_sum = n * (n + 1) / 2
            r_rb = float(1.0 - (2.0 * stat) / max(1.0, total_rank_sum))
            
        comparisons.append({
            "method": m,
            "baseline": baseline_method,
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "statistic": stat,
            "raw_p_value": p_val,
            "effect_size_rank_biserial": r_rb
        })
        
    # Apply Holm-Bonferroni correction
    # Sort by p_value ascending
    comparisons.sort(key=lambda x: x["raw_p_value"])
    m_count = len(comparisons)
    for idx, comp in enumerate(comparisons):
        adjusted_p = min(1.0, comp["raw_p_value"] * (m_count - idx))
        comp["adjusted_p_value"] = adjusted_p
        comp["is_significant_adj_05"] = bool(adjusted_p < 0.05)
        
    return {
        "baseline": baseline_method,
        "comparisons": comparisons
    }

def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Dict[str, Any]:
    """
    Computes McNemar's test for paired prediction-level comparison between two models.
    Contingency table:
      b: correct in A, incorrect in B
      c: incorrect in A, correct in B
    """
    corr_a = (preds_a == y_true)
    corr_b = (preds_b == y_true)
    
    # Discordant pairs
    b = int(np.sum(corr_a & (~corr_b))) # A correct, B wrong
    c = int(np.sum((~corr_a) & corr_b)) # A wrong, B correct
    
    if (b + c) == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        stat = float((abs(b - c) - 1.0) ** 2 / (b + c))
        p_val = float(1.0 - stats.chi2.cdf(stat, df=1))
        
    return {
        "test": "McNemar Test",
        "contingency_table": {"b_only_A_correct": b, "c_only_B_correct": c},
        "statistic": stat,
        "p_value": p_val,
        "is_significant_05": bool(p_val < 0.05)
    }
