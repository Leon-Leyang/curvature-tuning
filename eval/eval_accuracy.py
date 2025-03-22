"""
This file is for evaluating the accuracy metrics from the experiment logs.
"""
import re
import os
from collections import defaultdict
import statistics


def extract_accuracies_from_file(file_path, robustness=False):
    new_accuracies = defaultdict(list)
    baseline_accuracies = defaultdict(list)
    best_betas = defaultdict(list)

    if robustness:
        # Regex for robust accuracy
        pattern = (
            r"Best robust accuracy for ([\w\/\-\_]+) with ([\w]+) attack: "
            r"([0-9]+\.[0-9]+) with beta=([0-9\.]+), compared to ReLU accuracy: ([0-9]+\.[0-9]+)"
        )
    else:
        # Regex for standard accuracy
        pattern = r"Best accuracy for ([\w\/\-\_]+): ([0-9]+\.[0-9]+) with beta=([0-9\.]+)"
        baseline_pattern = r"ReLU accuracy: ([0-9]+\.[0-9]+)"

    with open(file_path, 'r') as file:
        for line in file:
            if robustness:
                # Match robust accuracy lines
                match = re.search(pattern, line)
                if match:
                    ds_name = match.group(1)
                    attack_type = match.group(2)
                    dataset_with_attack = f"{ds_name}_{attack_type}"
                    new_accuracy = float(match.group(3))
                    beta = float(match.group(4))
                    baseline_accuracy = float(match.group(5))

                    new_accuracies[dataset_with_attack].append(new_accuracy)
                    baseline_accuracies[dataset_with_attack].append(baseline_accuracy)
                    # Only include beta if new_accuracy is larger than baseline_accuracy
                    if new_accuracy > baseline_accuracy:
                        best_betas[dataset_with_attack].append(beta)
            else:
                # Match standard accuracy lines
                match = re.search(pattern, line)
                baseline_match = re.search(baseline_pattern, line)

                if match:
                    current_dataset = match.group(1)
                    new_accuracy = float(match.group(2))
                    beta = float(match.group(3))
                    new_accuracies[current_dataset].append(new_accuracy)
                    if baseline_match:
                        baseline_accuracy = float(baseline_match.group(1))
                        baseline_accuracies[current_dataset].append(baseline_accuracy)
                        # Only include beta if new_accuracy is larger than baseline_accuracy
                        if new_accuracy > baseline_accuracy:
                            best_betas[current_dataset].append(beta)

    return new_accuracies, baseline_accuracies, best_betas


def compute_statistics(log_files, robustness=False, exclude_datasets=None):
    """
    Computes statistics across the provided log files.

    :param log_files: List of paths to log files.
    :param robustness: Boolean indicating whether to parse robust accuracy lines.
    :param exclude_datasets: List of dataset names (keys) to exclude from computation.
    :return: Tuple containing statistics dictionaries and overall improvement/beta statistics.
    """
    combined_new_accuracies = defaultdict(list)
    combined_baseline_accuracies = defaultdict(list)
    combined_betas = defaultdict(list)

    for log_file in log_files:
        file_new_accuracies, file_baseline_accuracies, file_betas = extract_accuracies_from_file(
            log_file, robustness
        )
        for dataset, accuracies in file_new_accuracies.items():
            combined_new_accuracies[dataset].extend(accuracies)
        for dataset, accuracies in file_baseline_accuracies.items():
            combined_baseline_accuracies[dataset].extend(accuracies)
        for dataset, betas in file_betas.items():
            combined_betas[dataset].extend(betas)

    # Remove datasets in the exclusion list if provided
    if exclude_datasets:
        for dataset in list(combined_new_accuracies.keys()):
            if dataset in exclude_datasets:
                del combined_new_accuracies[dataset]
                if dataset in combined_baseline_accuracies:
                    del combined_baseline_accuracies[dataset]
                if dataset in combined_betas:
                    del combined_betas[dataset]

    # Compute mean and standard deviation for each dataset
    stats_new_accuracies = {
        dataset: (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0)
        for dataset, values in combined_new_accuracies.items()
    }
    stats_baseline_accuracies = {
        dataset: (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0)
        for dataset, values in combined_baseline_accuracies.items()
    }
    stats_betas = {
        dataset: (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0)
        for dataset, values in combined_betas.items()
    }

    # Compute normalized improvement and absolute improvement per dataset
    normalized_improvements = {}
    absolute_improvements = {}
    for dataset in stats_new_accuracies:
        baseline_mean = stats_baseline_accuracies.get(dataset, (0, 0))[0]
        if baseline_mean > 0:
            new_mean = stats_new_accuracies[dataset][0]
            normalized_improvements[dataset] = ((new_mean - baseline_mean) / baseline_mean) * 100
            absolute_improvements[dataset] = new_mean - baseline_mean

    # Compute overall improvements across all datasets
    overall_normalized_improvement = (
        sum(normalized_improvements.values()) / len(normalized_improvements)
        if normalized_improvements else 0
    )
    overall_absolute_improvement = (
        sum(absolute_improvements.values()) / len(absolute_improvements)
        if absolute_improvements else 0
    )

    # Compute overall beta as the average over the per-dataset mean beta values.
    # This is different from flattening all beta values across runs.
    beta_means = [mean for mean, _ in stats_betas.values() if mean is not None]
    if beta_means:
        overall_beta_mean = statistics.mean(beta_means)
        overall_beta_std = statistics.stdev(beta_means) if len(beta_means) > 1 else 0
    else:
        overall_beta_mean, overall_beta_std = 0, 0

    return (
        stats_new_accuracies, stats_baseline_accuracies, stats_betas,
        normalized_improvements, absolute_improvements,
        overall_normalized_improvement, overall_absolute_improvement,
        overall_beta_mean, overall_beta_std
    )


if __name__ == "__main__":
    # Input: List of log file paths
    log_files = [
        "../logs/post_replace_classification_replace_lp_coeff0.5_topk1_reg1_resnet18_seed42.log",
        "../logs/post_replace_classification_replace_lp_coeff0.5_topk1_reg1_resnet18_seed43.log",
        "../logs/post_replace_classification_replace_lp_coeff0.5_topk1_reg1_resnet18_seed44.log"
    ]

    # Define a list of dataset names to exclude (match the keys as extracted from the log file)
    exclude_datasets = [
    ]

    # Ensure files exist
    for file in log_files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            exit(1)

    # Compute statistics for standard accuracy, excluding specified datasets
    (
        stats_new_accuracies, stats_baseline_accuracies, stats_betas,
        normalized_improvements, absolute_improvements,
        overall_normalized_improvement, overall_absolute_improvement,
        overall_beta_mean, overall_beta_std
    ) = compute_statistics(log_files, robustness=False, exclude_datasets=exclude_datasets)

    print("Standard Accuracy, Beta Statistics, Normalized and Absolute Improvements:")
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
