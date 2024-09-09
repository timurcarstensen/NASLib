#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --mem 16000            # memory pool for all cores (4GB)
#SBATCH -t 00-10:00           # time (D-HH:MM)
#SBATCH -c 8                # number of cores
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o %x.%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e %x.%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH --mail-type=END,FAIL  # (recive mails about end and timeouts/crashes of your job)
#SBATCH -J naslib-predictor-eval              # sets the job name.

# Print some information about the job to STDOUT
#set -x

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config_file> <predictor> <trial>"
    exit 1
fi

config_file=$1
predictor=$2
trial=$3

base_file=/work/dlclarge1/carstent-timur_thesis_follow_up/NASLib

echo "Config file: $config_file"
echo "Predictor: $predictor"
echo "Trial: $trial"
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual env so that run_experiment can load the correct packages
source /work/dlclarge1/carstent-timur_thesis_follow_up/NASLib/.venv/bin/activate

python $base_file/benchmarks/predictors/runner.py --config-file $config_file

# Print some Information about the end-time to STDOUT
echo "DONE"
echo "Finished at $(date)"
