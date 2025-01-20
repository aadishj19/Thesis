# data_processing.py

"""
Main Analysis Script

This script reads TESS lightcurve FITS files, identifies flare candidates using ARIMA,
validates them based on shape and local variability criteria, and calculates physical properties of validated flares.

Functions:
- `validate_flare_shape`: Validates the shape of a potential flare based on symmetry, duration, and rise/decay rates.
- `check_local_variability`: Checks the local flux variability around a flare candidate to ensure significance.
- `analyze_tess_data_from_files`: Main function to process TESS data files, detect and analyze flares, and save results.

Usage:
1. Prepare input files:
   - TESS lightcurve FITS files to analyze.
   - TESS response function file (`trf_file`).
   - Pecaut & Mamajek table file (`pecaut_mamajek_file`).
2. Specify the output CSV file path (`output_file`).
3. Call `analyze_tess_data_from_files` with the file paths and output path.
"""

import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from astropy.constants import sigma_sb
from lightkurve import TessLightCurveFile
from astropy.io import fits
import contextlib
from tqdm import tqdm
from src.constants import R_sun_to_cm, BTJD_OFFSET, days_to_seconds
from src.utils import read_tess_response_function, read_pecaut_mamajek_table, convert_to_bjd
from src.lightcurve_analysis import planck_function, integrate_luminosity, fit_exponential_decay
from src.lightcurve_analysis import integrate_exponential_decay, find_flare_times, equivalent_duration

def validate_flare_shape(time, flux, start_idx, peak_idx, end_idx):
    """
    Validate flare shape characteristics with additional symmetry checks.
    """
    if not (0 <= start_idx < peak_idx < end_idx < len(flux)):
        return False
        
    rise_segment = flux[start_idx:peak_idx + 1]
    decay_segment = flux[peak_idx:end_idx + 1]
    
    if len(rise_segment) < 1 or len(decay_segment) < 1:
        return False
    
    # Basic trend checks
    rise_trend = rise_segment[-1] > rise_segment[0]
    decay_trend = decay_segment[-1] < decay_segment[0]
    if not (rise_trend and decay_trend):
        return False
    
    # Check if peak is significantly above the baseline
    pre_flare_level = np.median(flux[max(0, start_idx-5):start_idx])
    peak_height = flux[peak_idx] - pre_flare_level
    if peak_height <= 0:
        return False
        
    # New checks for symmetry and duration
    rise_time = time[peak_idx] - time[start_idx]
    decay_time = time[end_idx] - time[peak_idx]
    
    # Check for too-long duration (assuming time is in days)
    total_duration = time[end_idx] - time[start_idx]
    if total_duration > 0.2:  # > 2.4 hours might be too long
        return False
        
    # Check for too-symmetric timing
    symmetry_ratio = rise_time / decay_time
    if 0.5 < symmetry_ratio < 2.0:  # Too symmetrical
        return False
        
    # Check rise vs decay rates
    rise_rate = peak_height / rise_time
    decay_rate = peak_height / decay_time
    if decay_rate > rise_rate * 1.0:  # Decay should be significantly slower
        return False

    return True

def check_local_variability(flux, start_idx, end_idx, window=150):
    """
    Check if the local region around the flare shows excessive variability.
    """
    pre_flare = flux[max(0, start_idx - window):start_idx]
    post_flare = flux[end_idx:min(len(flux), end_idx + window)]
    
    if len(pre_flare) < 10 or len(post_flare) < 10:
        return False
        
    local_std = np.std(np.concatenate([pre_flare, post_flare]))
    flare_amplitude = np.max(flux[start_idx:end_idx]) - np.median(pre_flare)
    
    if local_std > flare_amplitude * 0.6:
        return False
    
    pre_trend = np.abs(np.mean(np.diff(pre_flare)))
    post_trend = np.abs(np.mean(np.diff(post_flare)))
    
    if pre_trend > local_std * 1.0 or post_trend > local_std * 1.0:
        return False
    
    return True

def analyze_tess_data_from_files(file_paths, trf_file, pecaut_mamajek_file, output_file):
    """
    Analyze TESS data files for flare detection with improved error handling and logging.
    """
    logging.info(f"Starting analysis of {len(file_paths)} files")
    logging.info(f"Output will be saved to: {output_file}")

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize headers and results list
    headers = ['TIC_ID', 'T_eff', 'Start_time', 'End_time', 'Peak_time(BJD)',
               'Amplitude', 'Duration(days)', 'Flare_energy(erg)', 'ED(s)']
    results = []
    
    try:
        # Read required data
        wavelengths, R_lambda = read_tess_response_function(trf_file)
        pecaut_mamajek_data = read_pecaut_mamajek_table(pecaut_mamajek_file)
    except Exception as e:
        logging.error(f"Failed to read required data files: {str(e)}")
        # Create empty output file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        return

    total_candidates = 0
    shape_filtered = 0
    variability_filtered = 0
    
    for file_path in file_paths:
        try:
            logging.info(f"Processing file: {file_path}")
            
            # Load light curve
            lcf = TessLightCurveFile(file_path)
            flux = lcf.flux.value
            time = lcf.time.value
            
            if flux.size == 0 or time.size == 0:
                logging.warning(f"Empty light curve data in {file_path}")
                continue
                
            # Normalize flux
            median_flux = np.median(flux)
            std_flux = np.std(flux)
            normalized_flux = (flux - median_flux) / std_flux + 1
            
            # Extract header information
            header = fits.getheader(file_path)
            teff = header.get('TEFF')
            radius_star = header.get('RADIUS')
            
            # Extract TIC ID
            file_name = os.path.basename(file_path)
            obs_id = file_name.split('-')[2]
            tic_id = obs_id.lstrip('0').split('-')[0]
            
            # Check for missing values
            teff_missing = teff is None or not np.isfinite(teff)
            radius_missing = radius_star is None or not np.isfinite(radius_star)
            radius_star_cm = radius_star * R_sun_to_cm if not radius_missing else None
            
            # Calculate stellar parameters
            B_lambda_star = None
            L_star_prime = None
            if not teff_missing:
                B_lambda_star = planck_function(wavelengths, teff)
                L_star_prime = integrate_luminosity(R_lambda, B_lambda_star, wavelengths)
            
            # ARIMA modeling
            warnings.filterwarnings("ignore")
            best_aic = float("inf")
            best_model = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(normalized_flux, order=(p, d, q))
                            model_fit = model.fit()
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_model = model_fit
                        except:
                            continue
            
            if best_model is None:
                logging.warning(f"Failed to fit ARIMA model for {file_path}")
                continue
                
            predicted_flux = best_model.predict(start=0, end=len(time) - 1)
            residuals = normalized_flux - predicted_flux
            
            # Flare detection (parameters vary depending on cadence)
            flare_threshold = np.std(residuals) * 3
            min_consecutive_points = 2
            min_amplitude = 0.0004
            max_duration = 0.8
            
            flare_indices = np.where(residuals > flare_threshold)[0]
            valid_flare_indices = []
            
            for idx in flare_indices:
                if (idx + min_consecutive_points - 1 < len(time) and 
                    np.all(residuals[idx:idx + min_consecutive_points] > flare_threshold)):
                    valid_flare_indices.append(idx)

            
            # Cluster flares
            min_distance = 2
            clusters = []
            current_cluster = []
            
            for index in valid_flare_indices:
                if len(current_cluster) == 0 or index - current_cluster[-1] <= min_distance:
                    current_cluster.append(index)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [index]
            
            if current_cluster:
                clusters.append(current_cluster)
            
            filtered_flare_indices = [cluster[np.argmax(residuals[cluster])] for cluster in clusters]

            # Find flare times
            flare_start_times, flare_end_times, flare_peak_times = find_flare_times(
                filtered_flare_indices, time, normalized_flux
            )
            
            # Process each flare
            for start_time, peak_time, end_time in zip(flare_start_times, flare_peak_times, flare_end_times):
                total_candidates += 1
                
                start_index = np.argmin(np.abs(time - start_time))
                peak_index = np.argmin(np.abs(time - peak_time))
                end_index = np.argmin(np.abs(time - end_time))
                
                if not validate_flare_shape(time, normalized_flux, start_index, peak_index, end_index):
                    shape_filtered += 1
                    continue
                    
                if not check_local_variability(normalized_flux, start_index, end_index):
                    variability_filtered += 1
                    continue
                
                # Calculate flare characteristics
                pre_flare_flux = np.median(normalized_flux[max(0, start_index-10):start_index])
                peak_flux = normalized_flux[peak_index]
                amplitude = (peak_flux - 1) * std_flux / median_flux
                duration = end_time - start_time
                decline_duration = end_time - peak_time
                rise_duration = peak_time - start_time
                points_between = len(np.where((time >= start_time) & (time <= end_time))[0])
                
                # Additional filtering criteria
                if (duration > max_duration or
                    amplitude < min_amplitude or
                    decline_duration <= rise_duration or
                    points_between < 2):
                    continue
                
                #peak_time_bjd = convert_to_bjd(peak_time, BTJD_OFFSET)
                peak_time_bjd = peak_time

                # Calculate flare energy.
                flare_energy = "N/A"
                if not (teff_missing or radius_missing):
                    try:
                        flare_time_segment = time[(time >= start_time) & (time <= end_time)]
                        flare_flux_segment = normalized_flux[(time >= start_time) & (time <= end_time)]
                        
                        if len(flare_time_segment) > 0:
                            initial_guess = (flare_flux_segment.max(), peak_time, 0.1, 1)
                            params = fit_exponential_decay(flare_time_segment, flare_flux_segment, initial_guess)
                            
                            A_flare = integrate_exponential_decay(params, start_time, end_time)
                            B_lambda_flare = planck_function(wavelengths, 9000)
                            L_flare_prime = integrate_luminosity(R_lambda, B_lambda_flare, wavelengths)
                            A_flare_abs = A_flare * np.pi * radius_star_cm ** 2 * L_star_prime / L_flare_prime
                            flare_luminosity = sigma_sb.value * (9000 ** 4) * A_flare_abs
                            flare_energy = flare_luminosity * (end_time - start_time) * days_to_seconds
                    except Exception as e:
                        logging.warning(f"Error calculating flare energy: {str(e)}")
                
                # Calculate equivalent duration
                try:
                    ed = equivalent_duration(time, normalized_flux, start_index, end_index)
                except Exception as e:
                    logging.warning(f"Error calculating equivalent duration: {str(e)}")
                    ed = np.nan
                
                results.append([
                    tic_id, teff, start_time, end_time, f"{peak_time_bjd:.6f}",
                    amplitude, duration, flare_energy, ed
                ])
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            continue
        
    # Save results
    if results:
        headers = ['TIC_ID', 'T_eff', 'Start_time', 'End_time', 'Peak_time(BJD)',
                  'Amplitude', 'Duration(days)', 'Flare_energy(erg)', 'ED(s)']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write results
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(results)

    # Log statistics
    logging.info(f"Total flare candidates: {total_candidates}")
    logging.info(f"Filtered by shape: {shape_filtered}")
    logging.info(f"Filtered by variability: {variability_filtered}")
    logging.info(f"Final flare count: {len(results)}")
