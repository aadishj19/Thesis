# scripts/process_chunk.py
import os
import sys
import yaml
import logging
import numpy as np
import fnmatch
from pathlib import Path
from datetime import datetime
from src.data_processing import analyze_tess_data_from_files

def setup_logging(output_dir, job_id):
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'job_{job_id}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    print("Argument count:", len(sys.argv))
    print("Arguments:", sys.argv)

    if len(sys.argv) != 4:
        print("Usage: python process_chunk.py <config_file> <chunk_index> <total_chunks>")
        sys.exit(1)

    config_file = sys.argv[1]
    chunk_idx = int(sys.argv[2])
    total_chunks = int(sys.argv[3])
    
    print("Config file:", config_file)
    print("Chunk index:", chunk_idx)
    print("Total chunks:", total_chunks)

    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("Config type:", type(config))
        print("Config contents:", config)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        sys.exit(1)

    if not isinstance(config, dict):
        print(f"Error: config is not a dictionary. Type: {type(config)}")
        sys.exit(1)

    if 'paths' not in config or 'output_dir' not in config['paths']:
        print("Error: config does not contain 'paths' -> 'output_dir'")
        sys.exit(1)

    # Get SLURM job ID
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # Setup logging
    setup_logging(config['paths']['output_dir'], job_id)
    
    # Get all input files
    data_dir = Path(config['paths']['data_dir'])
    all_files = sorted([
        str(f) for f in data_dir.glob('**/*-lc.fits')] + 
        [str(f) for f in data_dir.glob('**/*_lc.fits')]
    )
    
    # Calculate chunk boundaries
    chunk_size = len(all_files) // total_chunks
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < total_chunks - 1 else len(all_files)
    
    # Get files for this chunk
    chunk_files = all_files[start_idx:end_idx]
    
    logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}")
    logging.info(f"Number of files in chunk: {len(chunk_files)}")
    
    # Process files
    output_file = os.path.join(
        config['paths']['output_dir'],
        f'results_chunk_{chunk_idx}.csv'
    )
    
    try:
        analyze_tess_data_from_files(
            chunk_files,
            config['paths']['trf_file'],
            config['paths']['pecaut_mamajek_file'],
            output_file
        )
        logging.info(f"Successfully processed chunk {chunk_idx + 1}")
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
        raise

if __name__ == "__main__":
    main()
