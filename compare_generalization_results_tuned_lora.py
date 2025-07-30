import json
import os
import numpy as np


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    model_list = ['resnet18', 'swin_t']
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
    method_list = ['tuned_lora_rank1', 'tuned_lora_rank2', 'tuned_lora_rank4']
    seeds = [42]

    for model in model_list:
        pretrained_ds = 'imagenette' if 'swin' in model else 'imagenet'
        print('-' * 20)
        print(f'Comparing methods on {model}...')
        print('-' * 20)
        result_dict = {}
        valid_datasets = []

        # For accumulating overall stats
        overall_stats = {
            method: {
                'accuracies': [],
                'alpha_rank_ratios': []
            }
            for method in method_list
        }

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

            method_metrics = {m: {'accuracy': [], 'num_params': [], 'best_alpha_rank_ratio': []} for m in method_list}

            for seed in seeds:
                for method in method_list:
                    file_path = f'./results/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                    data = load_json(file_path)
                    method_metrics[method]['accuracy'].append(data['accuracy'])
                    method_metrics[method]['num_params'].append(data['num_params'])
                    method_metrics[method]['best_alpha_rank_ratio'].append(data['best_alpha_rank_ratio'])

            averaged_data = {}
            for method in method_list:
                accs = method_metrics[method]['accuracy']
                params = method_metrics[method]['num_params']
                best_alpha_rank_ratio = method_metrics[method]['best_alpha_rank_ratio']
                averaged_data[f"{method}_accuracy"] = np.mean(accs)
                averaged_data[f"{method}_accuracy_std"] = np.std(accs)
                averaged_data[f"{method}_num_params"] = np.mean(params)
                averaged_data[f"{method}_best_alpha_rank_ratio"] = np.mean(best_alpha_rank_ratio)
                averaged_data[f"{method}_best_alpha_rank_ratio_std"] = np.std(best_alpha_rank_ratio)

                # Accumulate for overall stats
                overall_stats[method]['accuracies'].append(averaged_data[f"{method}_accuracy"])
                overall_stats[method]['alpha_rank_ratios'].append(averaged_data[f"{method}_best_alpha_rank_ratio"])

            valid_datasets.append(transfer_ds)

            # Print per dataset stats
            print(f'[{transfer_ds}]')
            for method in method_list:
                acc_mean = averaged_data[f'{method}_accuracy']
                acc_std = averaged_data[f'{method}_accuracy_std']
                param_mean = averaged_data[f'{method}_num_params']
                best_alpha_rank_ratio_mean = averaged_data[f'{method}_best_alpha_rank_ratio']
                print(f"{method}: acc = {acc_mean:.2f} ± {acc_std:.2f}, params = {param_mean}, best alpha rank ratio {best_alpha_rank_ratio_mean:.2f}")
            print()

        # Print overall average across datasets
        print("====== Overall Averages ======")
        for method in method_list:
            accs = overall_stats[method]['accuracies']
            alphas = overall_stats[method]['alpha_rank_ratios']
            if accs:
                print(f"{method}:")
                print(f"  Accuracy: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
                print(f"  Best alpha rank ratio: {np.mean(alphas):.2f} ± {np.std(alphas):.2f}")
        print()
