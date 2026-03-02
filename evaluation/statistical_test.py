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
    # TODO
    raise NotImplementedError


def cohens_d(x, y):
    """
    Compute Cohen's d effect size for paired samples.

    Args:
        x: numpy array, scores from model A
        y: numpy array, scores from model B (same length as x)
    Returns:
        float, Cohen's d
    """
    # TODO standard Cohen's d for paired samples
    raise NotImplementedError


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
    # TODO
    raise NotImplementedError


def print_summary_table(results_list):
    """
    Pretty-print a summary table of statistical test results.

    Args:
        results_list: list of dicts from run_tests()
    """
    # TODO print a nice table with the results
    raise NotImplementedError


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
