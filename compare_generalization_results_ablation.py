import json
import os
import numpy as np


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    model_list = ['resnet18', 'resnet50']
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
    method_list = ['ct', 'ablation_swish_ct', 'ablation_softplus_ct']
    seeds = [42, 43, 44]

    for model in model_list:
        pretrained_ds = 'imagenette' if 'swin' in model else 'imagenet'
        print('-' * 20)
        print(f'Comparing methods on {model}...')
        print('-' * 20)
        result_dict = {}
        valid_datasets = []

        for transfer_ds in dataset_list:
            complete = True
            for method in method_list:
                for seed in seeds:
                    prefix = 'combined_search_ct' if method == 'ct' else method
                    file_path = f'./results/{prefix}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                    if not os.path.exists(file_path):
                        print(f'Missing: {file_path}')
                        complete = False
            if not complete:
                continue

            method_metrics = {
                m: {'accuracy': [], 'num_params': [], 'beta': []} for m in method_list
            }

            for seed in seeds:
                for method in method_list:
                    prefix = 'combined_search_ct' if method == 'ct' else method
                    file_path = f'./results/{prefix}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                    data = load_json(file_path)
                    method_metrics[method]['accuracy'].append(data['accuracy'])
                    method_metrics[method]['num_params'].append(data['num_params'])
                    if 'beta' in data:
                        method_metrics[method]['beta'].append(data['beta'])

            averaged_data = {}
            for method in method_list:
                accs = method_metrics[method]['accuracy']
                averaged_data[f"{method}_accuracy"] = np.mean(accs)
                averaged_data[f"{method}_accuracy_std"] = np.std(accs)

            ct_acc = averaged_data['ct_accuracy']
            result = {
                'ct_accuracy': ct_acc
            }

            for method in method_list:
                betas = method_metrics[method]['beta']
                if betas:
                    result[f'{method}_avg_beta'] = np.mean(betas)
                result[f'{method}_avg_accuracy'] = averaged_data[f"{method}_accuracy"]

                if method == 'ct':
                    continue

                result[method] = {
                    'rel_improve': (averaged_data[f"{method}_accuracy"] - ct_acc) / ct_acc,
                    'better_than_ct': averaged_data[f"{method}_accuracy"] > ct_acc
                }

            result_dict[transfer_ds] = result
            valid_datasets.append(transfer_ds)

            print(f'[{transfer_ds}]')
            for method in method_list:
                acc_mean = averaged_data[f"{method}_accuracy"]
                acc_std = averaged_data[f"{method}_accuracy_std"]
                print(f"{method}: acc = {acc_mean:.2f} ± {acc_std:.2f}")
                if f'{method}_avg_beta' in result:
                    print(f"{method} beta: {result[f'{method}_avg_beta']:.2f}")
            print()

        if valid_datasets:
            print(f"Summary for {model}:")
            for method in method_list:
                if method == 'ct':
                    continue
                rel_improvements = [result_dict[ds][method]['rel_improve'] for ds in valid_datasets]
                count_better = sum(result_dict[ds][method]['better_than_ct'] for ds in valid_datasets)
                beta_values = [result_dict[ds][f'{method}_avg_beta']
                               for ds in valid_datasets if f'{method}_avg_beta' in result_dict[ds]]
                acc_values = [result_dict[ds][f'{method}_avg_accuracy']
                              for ds in valid_datasets if f'{method}_avg_accuracy' in result_dict[ds]]

                print(f"{method}:")
                print(f"  Better than CT: {count_better} / {len(valid_datasets)}")
                print(f"  Relative improvement over CT: {100 * np.mean(rel_improvements):.2f}%")
                if acc_values:
                    print(f"  Average accuracy: {np.mean(acc_values):.2f}")
                if beta_values:
                    print(f"  Average beta: {np.mean(beta_values):.2f}")

            ct_accs = [result_dict[ds]['ct_accuracy'] for ds in valid_datasets]
            print(f"CT average accuracy: {np.mean(ct_accs):.2f}")

            ct_beta_values = [result_dict[ds]['ct_avg_beta']
                              for ds in valid_datasets if 'ct_avg_beta' in result_dict[ds]]
            if ct_beta_values:
                print(f"CT average beta: {np.mean(ct_beta_values):.2f}")
        else:
            print(f'No complete records for {model}.')