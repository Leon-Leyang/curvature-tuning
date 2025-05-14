import submitit
import os
from pathlib import Path
import numpy as np


def main(kwargs, job_dir):
    width = kwargs['width']
    depth = kwargs['depth']
    init_beta = kwargs['init_beta']

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
        slurm_partition="cs-all-gcondo", # Partition name
        slurm_job_name=f"demo_width{width}_depth{depth}_beta{init_beta:.2f}",  # Job name
        slurm_mail_type="ALL",    # Email settings
        slurm_mail_user="leyang_hu@brown.edu",  # Email address
    )

    command = (
        "module load miniconda3/23.11.0s && "
        "source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh && "
        "conda deactivate && "
        "conda activate spline && "
        f"python -u demo_classification_new.py --width {width} --depth {depth} --init_beta {init_beta:.2f}"
    )

    # Submit the job
    job = executor.submit(os.system, command)
    print(f"Job submitted for width {width}, depth {depth}, init_beta {init_beta:.2f}. Job ID: {job.job_id}")


if __name__ == "__main__":
    # Create the directory where logs and results will be saved
    job_dir = Path("./submitit")
    job_dir.mkdir(parents=True, exist_ok=True)

    width_list = list(range(5, 21))
    depth_list = [1]
    init_beta_list = list(np.arange(0.1, 1.0 - 1e-6, 0.1))

    for width in width_list:
        for depth in depth_list:
            for init_beta in init_beta_list:
                main({'width': width, 'depth': depth, 'init_beta': init_beta}, job_dir)
