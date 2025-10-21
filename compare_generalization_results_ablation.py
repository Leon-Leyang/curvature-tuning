import json
import os
import numpy as np


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    model_list = ['resnet18', 'resnet50', 'resnet152']
    dataset_list = [
        "arabic-characters",
        "arabic-digits",
        "beans",
        "cub200",
        "dtd",
        "fashion-mnist",
        "fgvc-aircraft",
        "flowers102",
        "food101",
        "medmnist/dermamnist",
        "medmnist/octmnist",
        "medmnist/pathmnist",
    ]
    method_list = ['ct', 'silu', 'softplus']
    seeds = [42, 43, 44]

    pretrained_ds = 'imagenet'

    for model in model_list:
        print('-' * 20)
        print(f'Comparing methods on {model}...')
        print('-' * 20)

        result_dict = {}
        valid_datasets = []

        for transfer_ds in dataset_list:
            complete = True
            for method in method_list:
                for seed in seeds:
                    file_path = f'./results/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                    if not os.path.exists(file_path):
                        print(f'Missing: {file_path}')
                        complete = False
            if not complete:
                continue

            # Collect metrics
            method_metrics = {m: {'accuracy': [], 'beta': []} for m in method_list}
            for seed in seeds:
                for method in method_list:
                    file_path = f'./results/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                    data = load_json(file_path)
                    method_metrics[method]['accuracy'].append(data['accuracy'])
                    if 'beta' in data:
                        method_metrics[method]['beta'].append(data['beta'])

            stats = {}
            for m in method_list:
                accs = method_metrics[m]['accuracy']
                betas = method_metrics[m]['beta']
                stats[m] = {
                    'acc_mean': float(np.mean(accs)),
                    'acc_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                    'beta_mean': float(np.mean(betas)) if betas else None,
                }

            ct_mean = stats['ct']['acc_mean']
            rel_to_ct = {}
            for m in method_list:
                if m == 'ct':
                    continue
                if ct_mean != 0:
                    rel_improve = (stats[m]['acc_mean'] - ct_mean) / ct_mean
                else:
                    rel_improve = float('inf') if stats[m]['acc_mean'] > 0 else 0.0
                rel_to_ct[m] = {
                    'rel_improve': rel_improve,
                    'better_than_ct': stats[m]['acc_mean'] > ct_mean
                }

            result = {'stats': stats, 'rel_to_ct': rel_to_ct}
            result_dict[transfer_ds] = result
            valid_datasets.append(transfer_ds)

            print(f'[{transfer_ds}]')
            for m in method_list:
                s = stats[m]
                beta_str = f" | beta={s['beta_mean']:.2f}" if s['beta_mean'] is not None else ""
                print(f"{m}: acc={s['acc_mean']:.2f} ± {s['acc_std']:.2f}{beta_str}")
            print()

        # Summary over datasets
        if valid_datasets:
            print(f"Summary for {model}:")
            # Per-method averages across datasets
            def collect(method, key):
                vals = []
                for ds in valid_datasets:
                    s = result_dict[ds]['stats'][method][key]
                    if s is not None:
                        vals.append(s)
                return vals

            # Non-CT summaries relative to CT
            for m in method_list:
                if m == 'ct':
                    continue
                rel_improvements = [result_dict[ds]['rel_to_ct'][m]['rel_improve'] for ds in valid_datasets]
                count_better = sum(result_dict[ds]['rel_to_ct'][m]['better_than_ct'] for ds in valid_datasets)
                acc_values = collect(m, 'acc_mean')
                beta_values = collect(m, 'beta_mean')

                print(f"{m}:")
                print(f"  Better than CT: {count_better} / {len(valid_datasets)}")
                print(f"  Relative improvement over CT: {100 * np.mean(rel_improvements):.2f}%")
                if acc_values:
                    print(f"  Average accuracy: {np.mean(acc_values):.2f}")
                if beta_values:
                    print(f"  Average beta: {np.mean(beta_values):.2f}")

            # CT summary
            ct_accs = [result_dict[ds]['stats']['ct']['acc_mean'] for ds in valid_datasets]
            print(f"CT average accuracy: {np.mean(ct_accs):.2f}")
            ct_betas = [b for b in collect('ct', 'beta_mean') if b is not None]
            if ct_betas:
                print(f"CT average beta: {np.mean(ct_betas):.2f}")
        else:
            print(f'No complete records for {model}.')
