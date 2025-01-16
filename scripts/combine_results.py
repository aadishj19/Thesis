import os
import sys
import pandas as pd
import yaml
import logging
from pathlib import Path

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'logs', 'combine_results.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: python combine_results.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    print(f"Loading config from: {config_file}")

    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Config loaded successfully. Type: {type(config)}")
        print(f"Config contents: {config}")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        sys.exit(1)

    if not isinstance(config, dict):
        print(f"Error: config is not a dictionary. Type: {type(config)}")
        sys.exit(1)

    if 'paths' not in config or 'output_dir' not in config['paths']:
        print("Error: config does not contain 'paths' -> 'output_dir'")
        sys.exit(1)

    output_dir = Path(config['paths']['output_dir'])
    setup_logging(output_dir)
    
    # Find all chunk results
    result_files = sorted(output_dir.glob('results_chunk_*.csv'))
    
    if not result_files:
        logging.error("No result files found!")
        return
    
    logging.info(f"Found {len(result_files)} result files to combine")
    
    # Combine all results
    dfs = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {str(e)}")
    
    if not dfs:
        logging.error("No valid data frames to combine!")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined results
    output_file = output_dir / 'combined_results.csv'
    combined_df.to_csv(output_file, index=False)
    logging.info(f"Combined results saved to {output_file}")
    
    # Clean up individual chunk files
    for file in result_files:
        try:
            file.unlink()
        except Exception as e:
            logging.error(f"Error deleting {file}: {str(e)}")

if __name__ == "__main__":
    main()