import json
import os


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


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
    method_list = ['base', 'ct', 'lora_rank1', 'search_ct']
    pretrained_ds = 'imagenet'
    seed = 42

    for model in model_list:
        print(f'Comparing methods on {model}...')
        result_dict = {}
        valid_datasets = []

        for transfer_ds in dataset_list:
            # Check for results completeness
            check_paths = [f'./results/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json' for method in method_list]
            if not all(os.path.exists(path) for path in check_paths):
                print(f'Missing record for {model} on {transfer_ds}.')
                continue

            # All methods expected to exist
            transfer_data = {}
            for method in method_list:
                file_path = f'./results/{method}_{pretrained_ds}_to_{transfer_ds.replace("/", "-")}_{model}_seed{seed}.json'
                transfer_data[method] = load_json(file_path)

            transfer_data['num_params_ratio'] = transfer_data['ct']['num_params'] / transfer_data['lora_rank1']['num_params']
            transfer_data['transfer_time_ratio'] = transfer_data['ct']['transfer_time'] / transfer_data['lora_rank1']['transfer_time']
            transfer_data['rel_improve_ct_to_base'] = (transfer_data['ct']['accuracy'] - transfer_data['base']['accuracy']) / transfer_data['base']['accuracy']
            transfer_data['rel_improve_ct_to_lora'] = (transfer_data['ct']['accuracy'] - transfer_data['lora_rank1']['accuracy']) / transfer_data['lora_rank1']['accuracy']
            transfer_data['ct_better_than_base'] = transfer_data['ct']['accuracy'] > transfer_data['base']['accuracy']
            transfer_data['ct_better_than_lora'] = transfer_data['ct']['accuracy'] > transfer_data['lora_rank1']['accuracy']
            transfer_data['rel_improve_search_ct_to_base'] = (transfer_data['search_ct']['accuracy'] - transfer_data['base']['accuracy']) / transfer_data['base']['accuracy']
            transfer_data['rel_improve_search_ct_to_lora'] = (transfer_data['search_ct']['accuracy'] - transfer_data['lora_rank1']['accuracy']) / transfer_data['lora_rank1']['accuracy']
            transfer_data['search_ct_better_than_base'] = transfer_data['search_ct']['accuracy'] > transfer_data['base']['accuracy']
            transfer_data['search_ct_better_than_lora'] = transfer_data['search_ct']['accuracy'] > transfer_data['lora_rank1']['accuracy']

            result_dict[transfer_ds] = transfer_data
            valid_datasets.append(transfer_ds)

        # Print aggregated statistics
        if valid_datasets:
            print(f'CT to LoRA num_params ratio for {model}: {sum([result_dict[ds]["num_params_ratio"] for ds in valid_datasets]) / len(valid_datasets):.4f}')
            print(f'CT to LoRA transfer_time ratio for {model}: {sum([result_dict[ds]["transfer_time_ratio"] for ds in valid_datasets]) / len(valid_datasets):.4f}')
            print(f'CT better than base for {model}: {sum([result_dict[ds]["ct_better_than_base"] for ds in valid_datasets])} out of {len(valid_datasets)} datasets.')
            print(f'CT to baseline relative improvement for {model}: {sum([result_dict[ds]["rel_improve_ct_to_base"] for ds in valid_datasets]) / len(valid_datasets):.4f}')
            print(f'CT better than LoRA for {model}: {sum([result_dict[ds]["ct_better_than_lora"] for ds in valid_datasets])} out of {len(valid_datasets)} datasets.')
            print(f'CT to LoRA relative improvement for {model}: {sum([result_dict[ds]["rel_improve_ct_to_lora"] for ds in valid_datasets]) / len(valid_datasets):.4f}')
            print(f'Search CT better than base for {model}: {sum([result_dict[ds]["search_ct_better_than_base"] for ds in valid_datasets])} out of {len(valid_datasets)} datasets.')
            print(f'Search CT to baseline relative improvement for {model}: {sum([result_dict[ds]["rel_improve_search_ct_to_base"] for ds in valid_datasets]) / len(valid_datasets):.4f}')
            print(f'Search CT better than LoRA for {model}: {sum([result_dict[ds]["search_ct_better_than_lora"] for ds in valid_datasets])} out of {len(valid_datasets)} datasets.')
            print(f'Search CT to LoRA relative improvement for {model}: {sum([result_dict[ds]["rel_improve_search_ct_to_lora"] for ds in valid_datasets]) / len(valid_datasets):.4f}')
        else:
            print(f'No complete records for {model}.')
