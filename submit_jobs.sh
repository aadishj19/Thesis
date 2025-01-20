#!/bin/bash
#SBATCH --account=username #make sure to add your slurm id here.
#SBATCH --job-name=jobname
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --array=0-99  # This will create 99 jobs
export PYTHONPATH="/path_to_file/:${PYTHONPATH}"
# Load  modules
ml anaconda
conda activate flarima

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Create directories
mkdir -p logs

# Run the processing script
python scripts/process_chunk.py /path_to_file/config.yaml ${SLURM_ARRAY_TASK_ID} 100

# When all array jobs complete, combine results
if [ "${SLURM_ARRAY_TASK_ID}" -eq "0" ]; then
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} combine_results.sh
fi
