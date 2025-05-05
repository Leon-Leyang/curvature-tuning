import submitit
import os
from pathlib import Path


def main(kwargs, job_dir):
    model = kwargs['model']
    pretrained_ds = kwargs['pretrained_ds']
    transfer_ds = kwargs['transfer_ds']
    seed = kwargs['seed']

    # Set up the executor folder to include the job ID placeholder
    executor = submitit.AutoExecutor(folder=job_dir / "%j")

    # Define SLURM parameters here, adjust based on your cluster's specifics
    executor.update_parameters(
        mem_gb=48,                # Memory allocation
        slurm_ntasks_per_node=1,  # Number of tasks per node
        cpus_per_task=6,          # Number of CPUs per task
        nodes=1,                  # Number of nodes
        timeout_min=2880,         # Maximum duration in minutes
        slurm_partition="cs-all-gcondo", # Partition name
        slurm_job_name=f"shared_ct_{pretrained_ds}_to_{transfer_ds}_{model}_seed{seed}",  # Job name
        slurm_mail_type="ALL",    # Email settings
        slurm_mail_user="leyang_hu@brown.edu",  # Email address
    )

    command = (
        "module load miniconda3/23.11.0s && "
        "source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh && "
        "conda deactivate && "
        "conda activate spline && "
        f"python -u classification_shared_ct.py --model {model} --pretrained_ds {pretrained_ds} --transfer_ds {transfer_ds} --seed {seed}"
    )

    # Submit the job
    job = executor.submit(os.system, command)
    print(f"Job submitted for model {model} on dataset {pretrained_ds}_to_{transfer_ds} with seed {seed} with job ID {job.job_id}")


def job_completed(pretrained_ds, transfer_ds, model, seed):
    transfer_ds_alias = transfer_ds.replace('/', '-')
    result_path = [
        f'./results/shared_ct_{pretrained_ds}_to_{transfer_ds_alias}_{model}_seed{seed}.json',
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

    pretrained_ds = 'imagenet'

    seed_list = [42, 43, 44]

    for seed in seed_list:
        for model in model_list:
            for transfer_ds in dataset_list:
                if not job_completed(pretrained_ds, transfer_ds, model, seed):
                    main({'model': model, 'pretrained_ds': pretrained_ds, 'transfer_ds': transfer_ds, 'seed': seed}, job_dir)
