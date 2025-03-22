"""
This file is for evaluating the robustness metrics from the experiment logs.
"""
import os
from eval_accuracy import compute_statistics


if __name__ == "__main__":
    # Input: List of log file paths
    log_files = [
        "../logs/post_replace_classification_lp_replace_coeff0.5_topk1_reg1_resnet18_seed42.log",
        "../logs/post_replace_classification_lp_replace_coeff0.5_topk1_reg1_resnet18_seed43.log",
        "../logs/post_replace_classification_lp_replace_coeff0.5_topk1_reg1_resnet18_seed44.log"
    ]

    # Define a list of dataset names to exclude (match the keys as extracted from the log file)
    exclude_datasets = [
    ]

    # Ensure files exist
    for file in log_files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            exit(1)

    # Compute statistics for robust accuracy, excluding specified datasets
    (
        stats_new_accuracies, stats_baseline_accuracies, stats_betas,
        normalized_improvements, absolute_improvements,
        overall_normalized_improvement, overall_absolute_improvement,
        overall_beta_mean, overall_beta_std
    ) = compute_statistics(log_files, robustness=True, exclude_datasets=exclude_datasets)

    print("\nRobust Accuracy, Beta Statistics, Normalized and Absolute Improvements:")
    for dataset in stats_new_accuracies:
        new_mean, new_std = stats_new_accuracies[dataset]
        baseline_mean, baseline_std = stats_baseline_accuracies.get(dataset, (0, 0))
        beta_mean, beta_std = stats_betas.get(dataset, (0, 0))
        normalized_improvement = normalized_improvements.get(dataset, 0)
        absolute_improvement = absolute_improvements.get(dataset, 0)
        print(f"Dataset: {dataset}")
        print(f"  Baseline Accuracy: {baseline_mean:.2f} ± {baseline_std:.2f}")
        print(f"  New Accuracy: {new_mean:.2f} ± {new_std:.2f}")
        print(f"  Beta (only if new > baseline): {beta_mean:.2f} ± {beta_std:.2f}")
        print(f"  Normalized Improvement: {normalized_improvement:.2f}%")
        print(f"  Absolute Improvement: {absolute_improvement:.2f}%\n")

    print(f"Overall Normalized Improvement Across All Datasets: {overall_normalized_improvement:.2f}%")
    print(f"Overall Absolute Improvement Across All Datasets: {overall_absolute_improvement:.2f}%")
    print(f"Overall Beta (average over dataset means): {overall_beta_mean:.2f} ± {overall_beta_std:.2f}")
