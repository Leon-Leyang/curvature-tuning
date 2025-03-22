"""
This file is for evaluating the per attribute metrics from the CelebA experiment logs.
"""
import re
import numpy as np


def parse_log_file(filename):
    """
    Parses one log file and returns a dictionary with keys 'acc', 'f1', and 'bal'.
    Each value is a list of groups (in order). Each group is the list of per-attribute
    float values extracted from the corresponding log line.

    It assumes that the three metric lines appear consecutively as:
      - a line containing "Per-attribute Accuracy:"
      - the next line containing "Per-attribute F1 Score:"
      - the following line containing "Per-attribute Balanced Accuracy:"
    """
    groups = {'acc': [], 'f1': [], 'bal': []}
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Look for the beginning of a group (an Accuracy line)
        if "Per-attribute Accuracy:" in lines[i]:
            # Make sure we have at least two more lines for f1 and balanced acc.
            if i + 2 < len(lines):
                acc_line = lines[i].strip()
                f1_line = lines[i + 1].strip()
                bal_line = lines[i + 2].strip()
                # Use regex to extract the comma-separated values.
                # The regex captures everything after the colon up to an optional "%" or the end of line.
                m_acc = re.search(r'Per-attribute Accuracy:\s*(.*?)(?:%|$)', acc_line)
                m_f1 = re.search(r'Per-attribute F1 Score:\s*(.*?)(?:%|$)', f1_line)
                m_bal = re.search(r'Per-attribute Balanced Accuracy:\s*(.*?)(?:%|$)', bal_line)
                if m_acc and m_f1 and m_bal:
                    try:
                        acc_vals = [float(x.strip()) for x in m_acc.group(1).split(',') if x.strip()]
                        f1_vals = [float(x.strip()) for x in m_f1.group(1).split(',') if x.strip()]
                        bal_vals = [float(x.strip()) for x in m_bal.group(1).split(',') if x.strip()]
                    except ValueError as ve:
                        print(f"Error parsing numbers in file {filename} at lines {i + 1}-{i + 3}: {ve}")
                        i += 3
                        continue
                    groups['acc'].append(acc_vals)
                    groups['f1'].append(f1_vals)
                    groups['bal'].append(bal_vals)
                    # Skip the next two lines since we already used them.
                    i += 3
                    continue
                else:
                    # If one of the regexes didn’t match, skip this line.
                    i += 1
            else:
                i += 1
        else:
            i += 1
    return groups


def main(log_files):
    # Define the 51 beta values. The ordering is:
    # beta[0] = 1.0, beta[1] = 0.5, then beta[2]=0.51, beta[3]=0.52, ..., beta[51-1]=0.99.
    beta_values = [1.0, 0.5] + [round(0.5 + i / 100, 2) for i in range(1, 50)]
    num_betas = len(beta_values)  # should be 51

    # Prepare a data structure for each metric.
    # For each metric, create a list of 51 sublists (one per beta) that will collect the
    # per-attribute vector (list of floats) from each log file (or run).
    data = {
        'acc': [[] for _ in range(num_betas)],
        'f1': [[] for _ in range(num_betas)],
        'bal': [[] for _ in range(num_betas)]
    }

    # Process each log file.
    for log_file in log_files:
        groups = parse_log_file(log_file)
        # Check that the file produced the expected number of groups.
        if (len(groups['acc']) != num_betas or
                len(groups['f1']) != num_betas or
                len(groups['bal']) != num_betas):
            print(f"Warning: In file '{log_file}' the number of groups is not {num_betas} as expected. "
                  f"Skipping this file. (Got {len(groups['acc'])} Accuracy groups, "
                  f"{len(groups['f1'])} F1 groups, {len(groups['bal'])} Balanced Accuracy groups.)")
            continue

        # For each beta index (i.e. group), append the extracted vector to the corresponding list.
        for i in range(num_betas):
            data['acc'][i].append(groups['acc'][i])
            data['f1'][i].append(groups['f1'][i])
            data['bal'][i].append(groups['bal'][i])

    # For each metric, compute the mean and std (over runs/log files) for each beta.
    results = {}
    for metric in ['acc', 'f1', 'bal']:
        means = []
        stds = []
        for beta_index in range(num_betas):
            runs = data[metric][beta_index]
            # Convert the list of lists to a NumPy array.
            # The shape is (num_runs, num_attributes)
            arr = np.array(runs)
            beta_mean = np.mean(arr, axis=0)  # mean per attribute
            beta_std = np.std(arr, axis=0)  # std per attribute
            means.append(beta_mean)
            stds.append(beta_std)
        # Convert to arrays of shape (num_betas, num_attributes)
        results[metric] = {'means': np.array(means), 'stds': np.array(stds)}

    # For each metric, determine for each attribute which beta gives the highest mean,
    # also note the metric value when using the overall best beta.
    # Finally, compute the average relative reduction when using the overall best beta
    # instead of the per-attribute best.
    num_attributes = results['acc']['means'].shape[1]
    for metric in ['acc', 'f1', 'bal']:
        # Determine overall best beta index for this metric.
        overall_means = np.mean(results[metric]['means'], axis=1)  # shape: (num_betas,)
        overall_best_idx = int(np.argmax(overall_means))
        overall_best_beta = beta_values[overall_best_idx]

        print(f"\nMetric: {metric.upper()}")
        print("Per-attribute best beta and metric values:")
        relative_reductions = []  # will store the relative reduction for each attribute

        for attr_idx in range(num_attributes):
            # For each attribute, find the beta with the best (highest) mean.
            col_means = results[metric]['means'][:, attr_idx]
            best_idx = int(np.argmax(col_means))
            best_beta = beta_values[best_idx]
            best_mean = results[metric]['means'][best_idx, attr_idx]
            best_std = results[metric]['stds'][best_idx, attr_idx]

            # Also get the metric value for the same attribute when using the overall best beta.
            overall_beta_mean = results[metric]['means'][overall_best_idx, attr_idx]
            overall_beta_std = results[metric]['stds'][overall_best_idx, attr_idx]

            # Compute relative reduction (if best_mean is nonzero).
            if best_mean != 0:
                rel_reduction = (best_mean - overall_beta_mean) / best_mean
            else:
                rel_reduction = 0.0
            relative_reductions.append(rel_reduction)

            print(f"  Attribute {attr_idx + 1:2d}:")
            print(f"    Best beta       = {best_beta}, {metric} = {best_mean:.2f} ± {best_std:.2f}")
            print(f"    Overall best beta ({overall_best_beta}) yields {metric} = "
                  f"{overall_beta_mean:.2f} ± {overall_beta_std:.2f}")

        # Compute the average relative reduction over all attributes (as a percentage).
        avg_relative_reduction = np.mean(relative_reductions) * 100

        # Also print the overall best beta's overall metric value (averaged over attributes).
        overall_mean = np.mean(results[metric]['means'][overall_best_idx, :])
        overall_std = np.mean(results[metric]['stds'][overall_best_idx, :])
        print(f"Overall best beta (averaged over attributes): beta = {overall_best_beta}, "
              f"{metric} = {overall_mean:.2f} ± {overall_std:.2f}")
        print(f"Average relative reduction (compared to per-attribute best): "
              f"{avg_relative_reduction:.2f}%")


if __name__ == '__main__':
    log_files = [
        "./logs/post_replace_classification_replace_lp_coeff0.5_topk1_reg1_resnet18_more_ds_seed42.log",
        "./logs/post_replace_classification_replace_lp_coeff0.5_topk1_reg1_resnet18_more_ds_seed43.log",
        "./logs/post_replace_classification_replace_lp_coeff0.5_topk1_reg1_resnet18_more_ds_seed44.log"
    ]
    main(log_files)
