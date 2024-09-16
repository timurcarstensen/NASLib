#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --mem 32000            # memory pool for all cores (4GB)
#SBATCH -t 00-10:00           # time (D-HH:MM)
#SBATCH -c 8                # number of cores
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o slurm_logs/%x.%A.%a.out       # STDOUT  (the folder log has to exist) %A is the job ID and %a is the array index
#SBATCH -e slurm_logs/%x.%A.%a.err       # STDERR  (the folder log has to exist) %A is the job ID and %a is the array index
#SBATCH --mail-type=END,FAIL  # (receive mails about end and timeouts/crashes of your job)
#SBATCH -J naslib-predictor-eval              # sets the job name.
#SBATCH --exclude=dlcgpu04,dlcgpu35,dlcgpu39  # exclude specific nodes

# Get the job parameters from the job array file
job_array_file=$1
job_params=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $job_array_file)

# Parse job parameters
read -r config_file predictor trial dataset <<< "$job_params"

base_file=/work/dlclarge1/carstent-timur_thesis_follow_up/NASLib/naslib

echo "Config file: $config_file"
echo "Predictor: $predictor"
echo "Trial: $trial"
echo "Dataset: $dataset"
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual env so that run_experiment can load the correct packages
source /work/dlclarge1/carstent-timur_thesis_follow_up/NASLib/.venv/bin/activate

python $base_file/benchmarks/predictors/runner.py --config-file $config_file

# Print some Information about the end-time to STDOUT
echo "DONE"
echo "Finished at $(date)"
