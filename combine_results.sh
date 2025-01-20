#!/bin/bash
#SBATCH --account=username #make sure to enter your slurm userid here.
#SBATCH --job-name=combine_results
#SBATCH --output=/path_to_file/FLARIMA/stdout.log
#SBATCH --error=/path_to_file/FLARIMA/stderr.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

# Load any required modules
ml anaconda
conda activate flarima

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the combine script
python scripts/combine_results.py /path_to_file/config.yaml
