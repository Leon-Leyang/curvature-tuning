import json
import os
import numpy as np


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    model_list = ['resnet18', 'resnet50', 'resnet152']
    dataset_list = ['cifar10', 'cifar100', 'imagenet']
    threat_list = ['Linf', 'L2', 'corruptions']
    method_list = ['base', 'ct']
    seeds = [42, 43, 44]
    n_examples = 1000

    for threat in threat_list:
        for model in model_list:
            print('-' * 30)
            print(f'Comparing methods on {model} under {threat} threat...')
            print('-' * 30)

            rel_improvements = []
            base_acc_all = []
            ct_acc_all = []
            beta_all = []

            for dataset in dataset_list:
                complete = True
                for method in method_list:
                    for seed in seeds:
                        file_path = f'./robust_results/{method}_{threat}_{dataset}_sample{n_examples}_{model}_seed{seed}.json'
                        if not os.path.exists(file_path):
                            print(f'Missing: {file_path}')
                            complete = False
                if not complete:
                    continue

                base_accs = []
                ct_accs = []
                beta_list = []

                for seed in seeds:
                    base_path = f'./robust_results/base_{threat}_{dataset}_sample{n_examples}_{model}_seed{seed}.json'
                    ct_path = f'./robust_results/ct_{threat}_{dataset}_sample{n_examples}_{model}_seed{seed}.json'
                    base_data = load_json(base_path)
                    ct_data = load_json(ct_path)

                    base_acc = base_data['accuracy']
                    ct_acc = ct_data['best_accuracy']
                    beta = ct_data.get('best_beta', 1.0)

                    if ct_acc < base_acc:
                        ct_acc = base_acc
                        beta = 1.0

                    base_accs.append(base_acc)
                    ct_accs.append(ct_acc)
                    beta_list.append(beta)

                avg_base_acc = np.mean(base_accs)
                std_base_acc = np.std(base_accs)
                avg_ct_acc = np.mean(ct_accs)
                std_ct_acc = np.std(ct_accs)
                avg_beta = np.mean(beta_list)
                std_beta = np.std(beta_list)

                print(f"[{dataset}]")
                print(f"base: acc = {avg_base_acc:.2f} ± {std_base_acc:.2f}")
                print(f"ct  : acc = {avg_ct_acc:.2f} ± {std_ct_acc:.2f}, beta = {avg_beta:.2f} ± {std_beta:.2f}")
                print()

                if avg_base_acc > 0:
                    rel_improve = (avg_ct_acc - avg_base_acc) / avg_base_acc
                    rel_improvements.append(rel_improve)

                base_acc_all.append(avg_base_acc)
                ct_acc_all.append(avg_ct_acc)
                beta_all.append(avg_beta)

            # Summary for this model + threat
            if base_acc_all:
                print(f"Avg base accuracy across datasets: {np.mean(base_acc_all):.2f}%")
                print(f"Avg CT accuracy   across datasets: {np.mean(ct_acc_all):.2f}%")
                print(f"Avg beta          across datasets: {np.mean(beta_all):.2f}")
                print(f"Avg rel. improvement of CT over base: {100 * np.mean(rel_improvements):.2f}%")
            else:
                print("No valid datasets to compute summary.")

            print()
