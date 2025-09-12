import os
import json
import numpy as np
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
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

    # (model, dataset, threat) -> method -> list of accs
    results = defaultdict(lambda: defaultdict(list))

    # dataset -> {'ct': [...], 'lora': [...]}
    dataset_improvements = defaultdict(lambda: {'ct': [], 'lora': []})

    for model in model_list:
        pretrained_ds = 'imagenet'
        for dataset in dataset_list:
            dataset_key = f"{pretrained_ds}_to_{dataset}"
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

    # (model, threat) -> method -> list of mean accs over datasets
    aggregate_summary = defaultdict(lambda: defaultdict(list))

    for (model, dataset, threat), method_accs in results.items():
        print(f"\n[{model} | {dataset} | {threat}]")
        means = {}
        for method in methods:
            if method in method_accs and method_accs[method]:
                accs = method_accs[method]
                mean = np.mean(accs)
                std = np.std(accs)
                means[method] = mean
                print(f"{method_display_names[method]}: acc = {mean:.2f} ± {std:.2f}")
                aggregate_summary[(model, threat)][method].append(mean)

        # Per-dataset relative improvements
        if 'base' in means:
            base = means['base']
            if base != 0:
                if 'train_ct' in means:
                    rel_ct = 100 * (means['train_ct'] - base) / base
                    dataset_improvements[dataset]['ct'].append(rel_ct)
                if 'lora_rank1_alpha1' in means:
                    rel_lora = 100 * (means['lora_rank1_alpha1'] - base) / base
                    dataset_improvements[dataset]['lora'].append(rel_lora)

    # Summary over datasets for each (model, threat)
    print("\n========== Summary by (Model, Threat) ==========")
    for (model, threat), method_means in aggregate_summary.items():
        print(f"\n[{model} | {threat}]")
        for method in methods:
            if method in method_means:
                avg = np.mean(method_means[method])
                print(f"{method_display_names[method]} avg acc over datasets: {avg:.2f}")

        if 'base' in method_means:
            base = np.mean(method_means['base'])
            if base != 0:
                if 'train_ct' in method_means:
                    ct = np.mean(method_means['train_ct'])
                    rel_ct = 100 * (ct - base) / base
                    print(f"Trainable CT rel. improvement over baseline: {rel_ct:.2f}%")
                if 'lora_rank1_alpha1' in method_means:
                    lora = np.mean(method_means['lora_rank1_alpha1'])
                    rel_lora = 100 * (lora - base) / base
                    print(f"LoRA rel. improvement over baseline: {rel_lora:.2f}%")
