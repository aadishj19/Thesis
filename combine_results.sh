#!/bin/bash
#SBATCH --account=username
#SBATCH --job-name=combine_results
#SBATCH --output=/STER/aadishj/Documents/FLARIMA/stdout.log
#SBATCH --error=/STER/aadishj/Documents/FLARIMA/stderr.log
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
