import json
import os
import numpy as np


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    model_list = ['resnet18', 'resnet50', 'resnet152', 'swin_t', 'swin_s']
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
    method_list = ['base', 'lora_rank1', 'search_ct', 'ct']
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
                    file_path = f'./results/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                    if not os.path.exists(file_path):
                        print(f'Missing: {file_path}')
                        complete = False
            if not complete:
                continue

            method_metrics = {m: {'accuracy': [], 'num_params': []} for m in method_list}

            for seed in seeds:
                for method in method_list:
                    file_path = f'./results/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                    data = load_json(file_path)
                    method_metrics[method]['accuracy'].append(data['accuracy'])
                    method_metrics[method]['num_params'].append(data['num_params'])

            # Compute mean and std
            averaged_data = {}
            for method in method_list:
                accs = method_metrics[method]['accuracy']
                params = method_metrics[method]['num_params']
                averaged_data[f"{method}_accuracy"] = np.mean(accs)
                averaged_data[f"{method}_accuracy_std"] = np.std(accs)
                averaged_data[f"{method}_num_params"] = np.mean(params)
                averaged_data[f"{method}_num_params_std"] = np.std(params)

            result = {
                'num_params_ratio': averaged_data['ct_num_params'] / averaged_data['lora_rank1_num_params'],
                'rel_improve_ct_to_base': (averaged_data['ct_accuracy'] - averaged_data['base_accuracy']) / averaged_data['base_accuracy'],
                'rel_improve_ct_to_lora': (averaged_data['ct_accuracy'] - averaged_data['lora_rank1_accuracy']) / averaged_data['lora_rank1_accuracy'],
                'ct_better_than_base': averaged_data['ct_accuracy'] > averaged_data['base_accuracy'],
                'ct_better_than_lora': averaged_data['ct_accuracy'] > averaged_data['lora_rank1_accuracy'],
                'rel_improve_search_ct_to_base': (averaged_data['search_ct_accuracy'] - averaged_data['base_accuracy']) / averaged_data['base_accuracy'],
                'rel_improve_search_ct_to_lora': (averaged_data['search_ct_accuracy'] - averaged_data['lora_rank1_accuracy']) / averaged_data['lora_rank1_accuracy'],
                'search_ct_better_than_base': averaged_data['search_ct_accuracy'] > averaged_data['base_accuracy'],
                'search_ct_better_than_lora': averaged_data['search_ct_accuracy'] > averaged_data['lora_rank1_accuracy'],
            }

            result_dict[transfer_ds] = result
            valid_datasets.append(transfer_ds)

            # Print per dataset stats
            print(f'[{transfer_ds}]')
            for method in method_list:
                acc_mean = averaged_data[f'{method}_accuracy']
                acc_std = averaged_data[f'{method}_accuracy_std']
                param_mean = averaged_data[f'{method}_num_params']
                print(f"{method}: acc = {acc_mean:.2f} ± {acc_std:.2f}, params = {param_mean}")
            print()

        if valid_datasets:
            print(f'CT to LoRA num_params ratio: {100 * sum([result_dict[ds]["num_params_ratio"] for ds in valid_datasets]) / len(valid_datasets):.2f}%')
            print(f'CT better than base: {sum([result_dict[ds]["ct_better_than_base"] for ds in valid_datasets])} / {len(valid_datasets)}')
            print(f'CT to base rel. improvement: {100 * sum([result_dict[ds]["rel_improve_ct_to_base"] for ds in valid_datasets]) / len(valid_datasets):.2f}%')
            print(f'CT better than LoRA: {sum([result_dict[ds]["ct_better_than_lora"] for ds in valid_datasets])} / {len(valid_datasets)}')
            print(f'CT to LoRA rel. improvement: {100 * sum([result_dict[ds]["rel_improve_ct_to_lora"] for ds in valid_datasets]) / len(valid_datasets):.2f}%')
            print(f'Search CT better than base: {sum([result_dict[ds]["search_ct_better_than_base"] for ds in valid_datasets])} / {len(valid_datasets)}')
            print(f'Search CT to base rel. improvement: {100 * sum([result_dict[ds]["rel_improve_search_ct_to_base"] for ds in valid_datasets]) / len(valid_datasets):.2f}%')
            print(f'Search CT better than LoRA: {sum([result_dict[ds]["search_ct_better_than_lora"] for ds in valid_datasets])} / {len(valid_datasets)}')
            print(f'Search CT to LoRA rel. improvement: {100 * sum([result_dict[ds]["rel_improve_search_ct_to_lora"] for ds in valid_datasets]) / len(valid_datasets):.2f}%')
        else:
            print(f'No complete records for {model}.')
