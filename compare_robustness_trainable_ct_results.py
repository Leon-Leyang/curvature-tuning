import os
import json
import numpy as np
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # -------------------- Config --------------------
    model_list = ['resnet18', 'resnet50', 'resnet152']
    dataset_list = ['cifar10', 'cifar100']
    threats = ['Linf', 'L2', 'corruptions']
    seeds = [42, 43, 44]

    methods = ['base', 'train_ct', 'lora_rank1_alpha1']
    method_display_names = {
        'base': 'Baseline',
        'train_ct': 'Trainable CT',
        'lora_rank1_alpha1': 'LoRA'
    }

    result_root = "./robust_results"

    # (model, dataset, threat) -> method -> list of accs (over seeds)
    results = defaultdict(lambda: defaultdict(list))

    # -------------------- Ingest --------------------
    for model in model_list:
        for dataset in dataset_list:
            dataset_key = f"imagenet_to_{dataset}"
            for threat in threats:
                key = (model, dataset, threat)
                for seed in seeds:
                    for method in methods:
                        file_name = f"{method}_{threat.lower()}_{dataset_key}_sample1000_{model}_seed{seed}.json"
                        file_path = os.path.join(result_root, file_name)
                        if not os.path.exists(file_path):
                            print(f"[Missing] {file_path}")
                            continue
                        try:
                            data = load_json(file_path)
                            acc = data['accuracy']
                            results[key][method].append(acc)
                        except Exception as e:
                            print(f"[Error] reading {file_path}: {e}")

    # -------------------- Per-(model,dataset,threat) summary and collection --------------------
    # For computing the metric, we need per-(model, threat, dataset) means
    # (model, threat) -> dataset -> method -> mean_acc_over_seeds
    per_dataset_means = defaultdict(lambda: defaultdict(dict))

    for (model, dataset, threat), method_accs in results.items():
        print(f"\n[{model} | {dataset} | {threat}]")
        means = {}
        for method in methods:
            if method in method_accs and method_accs[method]:
                accs = method_accs[method]
                mean = float(np.mean(accs))
                std = float(np.std(accs))
                means[method] = mean
                print(f"{method_display_names[method]}: acc = {mean:.2f} ± {std:.2f}")
                per_dataset_means[(model, threat)][dataset][method] = mean

    # -------------------- Summary by (model, threat) --------------------
    print("\n========== Summary by (Model, Threat) ==========")
    for (model, threat), dataset_to_methods in per_dataset_means.items():
        print(f"\n[{model} | {threat}]")

        # 1) Average accuracies across datasets (for reference)
        for method in methods:
            vals = [m[method] for m in dataset_to_methods.values() if method in m]
            if vals:
                print(f"{method_display_names[method]} avg acc over datasets: {np.mean(vals):.2f}")

        # 2) Average of per-dataset relative improvements
        #    Each dataset contributes equally; we compute relative improvement within each dataset, then average.
        #    Skip datasets with missing base or zero base.
        # Trainable CT to base
        ct_to_base_rels = []
        for ds, m in dataset_to_methods.items():
            if 'base' in m and m['base'] != 0 and 'train_ct' in m:
                ct_to_base_rels.append(100.0 * (m['train_ct'] - m['base']) / m['base'])
        if ct_to_base_rels:
            print(f"Trainable CT avg REL improvement: {np.mean(ct_to_base_rels):.2f}%")

        # LoRA to base
        lora_to_base_rels = []
        for ds, m in dataset_to_methods.items():
            if 'base' in m and m['base'] != 0 and 'lora_rank1_alpha1' in m:
                lora_to_base_rels.append(100.0 * (m['lora_rank1_alpha1'] - m['base']) / m['base'])
        if lora_to_base_rels:
            print(f"LoRA avg REL improvement: {np.mean(lora_to_base_rels):.2f}%")

        # Trainable CT to LoRA
        ct_to_lora_rels = []
        for ds, m in dataset_to_methods.items():
            if 'lora_rank1_alpha1' in m and m['lora_rank1_alpha1'] != 0 and 'train_ct' in m:
                ct_to_lora_rels.append(100.0 * (m['train_ct'] - m['lora_rank1_alpha1']) / m['lora_rank1_alpha1'])
        if ct_to_lora_rels:
            print(f"Trainable CT to LoRA avg REL improvement: {np.mean(ct_to_lora_rels):.2f}%")
