import submitit
import os
from pathlib import Path


def main(kwargs, job_dir):
    model = kwargs['model']
    dataset = kwargs['dataset']
    threat = kwargs['threat']
    batch_size = kwargs['batch_size']
    seed = kwargs['seed']

    # Set up the executor folder to include the job ID placeholder
    executor = submitit.AutoExecutor(folder=job_dir / "%j")

    # Define SLURM parameters here, adjust based on your cluster's specifics
    executor.update_parameters(
        mem_gb=48,                # Memory allocation
        slurm_ntasks_per_node=1,  # Number of tasks per node
        cpus_per_task=6,          # Number of CPUs per task
        gpus_per_node=1,          # Number of GPUs to use
        nodes=1,                  # Number of nodes
        timeout_min=5760,         # Maximum duration in minutes
        slurm_partition="gpu", # Partition name
        slurm_job_name=f"{threat}_{dataset}_{model}_seed{seed}",  # Job name
        slurm_mail_type="ALL",    # Email settings
        slurm_mail_user="leyang_hu@brown.edu",  # Email address
    )

    command = (
        "module load miniconda3/23.11.0s && "
        "source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh && "
        "conda deactivate && "
        "conda activate spline && "
        f"python -u robustness.py --model {model} --threat {threat} --dataset {dataset} --batch_size {batch_size} --seed {seed}"
    )

    # Submit the job
    job = executor.submit(os.system, command)
    print(f"Job submitted for {threat} threat on model {model} on dataset {dataset} with seed {seed} with job ID {job.job_id}")


def job_completed(threat, dataset, model, seed):
    result_path = [
        f'./robust_results/base_{threat}_{dataset}_{model}_seed{seed}.json',
        f'./robust_results/ct_{threat}_{dataset}_{model}_seed{seed}.json',
    ]

    # Check if all result files exist
    for path in result_path:
        if not os.path.exists(path):
            return False
    return True


if __name__ == "__main__":
    # Create the directory where logs and results will be saved
    job_dir = Path("./submitit")
    job_dir.mkdir(parents=True, exist_ok=True)

    # List of models
    model_list = [
        "resnet18",
        "resnet50",
        "resnet152",
    ]

    # List of datasets to transfer to
    dataset_list = [
        "cifar10",
        "cifar100",
        "imagenet"
    ]

    # List of threats
    threat_list = [
        "Linf",
        "L2",
        "corruptions",
    ]

    seed_list = [42, 43, 44]

    for seed in seed_list:
        for threat in threat_list:
            for model in model_list:
                for dataset in dataset_list:
                    batch_size = 1000 if dataset != "imagenet" else 250
                    if not job_completed(threat, dataset, model, seed):
                        main({'model': model, 'dataset': dataset, 'threat': threat, 'batch_size': batch_size, 'seed': seed}, job_dir)
