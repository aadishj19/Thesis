# FLARIMA: Flare Detection and Analysis with ARIMA Models

This repository is a toolset designed to process and analyze TESS data to detect and characterize flares. The package uses ARIMA models to process light curves.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Structure](#structure)
- [Scripts](#scripts)

## Installation

To prepare your environment, ensure you have Python and Anaconda installed, and follow these steps:

```bash
git clone https://github.com/yourusername/flarima.git
cd flarima
conda create --name flarima python=3.8
conda activate flarima
pip install -r requirements.txt
```

## Usage

It is primarily intended to be used via SLURM jobs to process large datasets. Here is a basic rundown of how to run the processing pipeline:

1. Update the paths in `config.yaml` to point to your data files and directory as well as mention the amount of cores you want to use.
2. Submit the batch job to SLURM:
   ```bash
   sbatch run_processing.sh
   ```

## Configuration

The configuration file, `config.yaml`, defines paths and job control parameters:

```yaml
paths:
  data_dir: "/path/to/data_dir/"
  output_dir: "/path/to/output_dir/" #this is where all your results will be saved.
  trf_file: "/path/to/trf_file.csv" #this is the TESS response function
  pecaut_mamajek_file: "/path/to/PecautMamajek2013.txt" #The pecaut_mamajek table can be found here: http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

processing:
  chunk_size: 125
  min_memory_per_job: "2GB"
  max_runtime: "24:00:00"
```

## Structure

The repository is structured as follows:

- `/scripts`: Contains scripts for job execution.
  - `process_chunk.py`: Processes individual data chunks.
  - `combine_results.py`: Combines results from individual chunks.
- `/src`: Contains source code for data processing and analysis.
  - `constants.py`: Defines physical constants used across the project.
  - `data_processing.py`: Handles the main data loading and preprocessing.
  - `lightcurve_analysis.py`: Functions for flare analysis using light curves.
  - `utils.py`: Utility functions for reading data and estimating parameters.

## Scripts

### run_processing.sh

A SLURM script to batch process data in chunks, which uses `process_chunk.py` and `combine_results.py`.

### process_chunk.py

Performs data processing tasks on a specified chunk of data. Calls functions from `src.data_processing` to analyze light curves. (A chunk in this context is a subset of data: for example if you use 99 cores for 16000 lightcurves then a chunk will be 16000/99.)

### combine_results.py

After job completion, it combines the results from multiple chunks into a single CSV file for easier analysis and interpretation.

