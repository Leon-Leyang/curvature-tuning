import os
import re
import statistics
from collections import defaultdict

def extract_all_final_accuracies(filepath):
    dataset_accuracies = []

    dataset_pattern = re.compile(r"Transfer learning from ([\w_]+) to ([\w_]+)")
    final_accuracy_pattern = re.compile(r"train:test_epoch:81 - Loss: [\d\.]+, Accuracy: ([\d\.]+)%")

    current_dataset = None
    current_final_accuracy = None

    with open(filepath, 'r') as f:
        for line in f:
            dataset_match = dataset_pattern.search(line)
            if dataset_match:
                # Save the previous dataset block (if any)
                if current_dataset and current_final_accuracy is not None:
                    dataset_accuracies.append((current_dataset, current_final_accuracy))
                # Start new block
                src, dst = dataset_match.groups()
                current_dataset = f"{src}_to_{dst}"
                current_final_accuracy = None  # Reset for the new block

            acc_match = final_accuracy_pattern.search(line)
            if acc_match:
                current_final_accuracy = float(acc_match.group(1))

    # Add the last block if valid
    if current_dataset and current_final_accuracy is not None:
        dataset_accuracies.append((current_dataset, current_final_accuracy))

    return dataset_accuracies


def compute_final_accuracy_stats_from_multiple_logs(log_files):
    dataset_to_accuracies = defaultdict(list)

    for file in log_files:
        if not os.path.exists(file):
            print(f"Warning: Log file not found: {file}")
            continue

        results = extract_all_final_accuracies(file)
        if not results:
            print(f"Warning: No final accuracies found in {file}")
        for dataset, acc in results:
            dataset_to_accuracies[dataset].append(acc)

    for dataset, accuracies in dataset_to_accuracies.items():
        mean_acc = statistics.mean(accuracies)
        std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        print(f"Dataset: {dataset}")
        print(f"  Final Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n")


if __name__ == "__main__":
    log_files = [
        "../logs/classification_lora_train_percentage1.0_rank1_alpha1.0_epoch30_resnet18_seed42.log",
        "../logs/classification_lora_train_percentage1.0_rank1_alpha1.0_epoch30_resnet18_seed43.log",
        "../logs/classification_lora_train_percentage1.0_rank1_alpha1.0_epoch30_resnet18_seed44.log"
    ]

    compute_final_accuracy_stats_from_multiple_logs(log_files)
