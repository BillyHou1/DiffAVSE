# Shunjie
# Statistical significance tests.
# Compare per-sample PESQ/STOI/SI-SDR scores between our model and a baseline.
# Run paired t-test, Wilcoxon signed-rank test, and compute Cohen's d effect size.
# Read scores from CSV, print results as a summary table.

import argparse
import csv
import numpy as np
from scipy import stats


def load_scores(csv_path):
    """
    Load per-sample metric scores from a CSV file.

    Expected CSV format (with header):
        sample_id, pesq, stoi, si_sdr
        sample_001, 2.31, 0.87, 12.5
        ...

    Args:
        csv_path: str
    Returns:
        dict with keys 'pesq', 'stoi', 'si_sdr', each mapping to a numpy array
    """
    scores = {'pesq': [], 'stoi': [], 'si_sdr': []}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores['pesq'].append(float(row['pesq']))
            scores['stoi'].append(float(row['stoi']))
            scores['si_sdr'].append(float(row['si_sdr']))
    
    return {k: np.array(v) for k, v in scores.items()}


def cohens_d(x, y):
    """
    Compute Cohen's d effect size for paired samples.

    Args:
        x: numpy array, scores from model A
        y: numpy array, scores from model B (same length as x)
    Returns:
        float, Cohen's d
    """
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)  # Use sample std
    
    # Avoid division by zero
    if std_diff < 1e-10:
        return 0.0
    
    return mean_diff / std_diff


def run_tests(scores_ours, scores_baseline, metric_name):
    """
    Run paired t-test, Wilcoxon signed-rank test, and compute Cohen's d.

    Args:
        scores_ours:     numpy array, per-sample scores from our model
        scores_baseline: numpy array, per-sample scores from baseline
        metric_name:     str, for display (e.g. 'PESQ')
    Returns:
        dict with keys:
            'metric':    str
            'mean_ours': float
            'mean_base': float
            't_stat':    float
            't_pvalue':  float
            'w_stat':    float
            'w_pvalue':  float
            'cohens_d':  float

    scipy.stats has ttest_rel and wilcoxon for this.
    """
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(scores_ours, scores_baseline)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, w_pvalue = stats.wilcoxon(scores_ours, scores_baseline)
    except ValueError:
        # Wilcoxon fails if all differences are zero
        w_stat, w_pvalue = 0.0, 1.0
    
    # Cohen's d
    d = cohens_d(scores_ours, scores_baseline)
    
    return {
        'metric': metric_name,
        'mean_ours': np.mean(scores_ours),
        'mean_base': np.mean(scores_baseline),
        't_stat': t_stat,
        't_pvalue': t_pvalue,
        'w_stat': w_stat,
        'w_pvalue': w_pvalue,
        'cohens_d': d
    }


def print_summary_table(results_list):
    """
    Pretty-print a summary table of statistical test results.

    Args:
        results_list: list of dicts from run_tests()
    """
    print("\n" + "="*100)
    print("STATISTICAL SIGNIFICANCE TEST RESULTS")
    print("="*100)
    print(f"{'Metric':<8} {'Mean(Ours)':<12} {'Mean(Base)':<12} {'t-stat':<10} {'t-p':<10} {'W-stat':<10} {'W-p':<10} {'Cohen d':<10}")
    print("-"*100)
    
    for r in results_list:
        metric = r['metric']
        mean_ours = r['mean_ours']
        mean_base = r['mean_base']
        t_stat = r['t_stat']
        t_p = r['t_pvalue']
        w_stat = r['w_stat']
        w_p = r['w_pvalue']
        d = r['cohens_d']
        
        # Mark significant p-values
        t_sig = '*' if t_p < 0.05 else ''
        w_sig = '*' if w_p < 0.05 else ''
        
        print(f"{metric:<8} {mean_ours:<12.4f} {mean_base:<12.4f} {t_stat:<10.4f} {t_p:<9.4f}{t_sig} {w_stat:<10.4f} {w_p:<9.4f}{w_sig} {d:<10.4f}")
    
    print("="*100)
    print("Note: * indicates p < 0.05 (statistically significant)")
    
    # Interpret Cohen's d
    print("\nEffect size interpretation (Cohen's d):")
    print("  |d| < 0.2:  negligible")
    print("  0.2 <= |d| < 0.5: small")
    print("  0.5 <= |d| < 0.8: medium")
    print("  |d| >= 0.8: large")


def main():
    parser = argparse.ArgumentParser(description='Statistical significance tests')
    parser.add_argument('--ours_csv', required=True,
                        help='CSV with per-sample scores from our model')
    parser.add_argument('--baseline_csv', required=True,
                        help='CSV with per-sample scores from baseline model')
    args = parser.parse_args()

    scores_ours = load_scores(args.ours_csv)
    scores_baseline = load_scores(args.baseline_csv)

    metrics = ['pesq', 'stoi', 'si_sdr']
    results = []
    for m in metrics:
        r = run_tests(scores_ours[m], scores_baseline[m], m.upper())
        results.append(r)

    print_summary_table(results)


if __name__ == '__main__':
    main()
