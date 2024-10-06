import os
import h5py
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Define a function to process a single file
def process_file(file_path):
    with h5py.File(file_path, "r") as file:
        data = file["data"][:]
        results = {
            "mean": data.mean(axis=(1, 2)).tolist(),
            "std": np.std(data, axis=(1, 2)).tolist(),
            "max": data.max(axis=(1, 2)).tolist(),
            "min": data.min(axis=(1, 2)).tolist(),
            "median": np.median(data, axis=(1, 2)).tolist(),
            "percentile_25": np.percentile(data, 25, axis=(1, 2)).tolist(),
            "percentile_75": np.percentile(data, 75, axis=(1, 2)).tolist(),
            "percentile_5": np.percentile(data, 5, axis=(1, 2)).tolist(),
            "percentile_95": np.percentile(data, 95, axis=(1, 2)).tolist(),
            "percentile_1": np.percentile(data, 1, axis=(1, 2)).tolist(),
            "percentile_99": np.percentile(data, 99, axis=(1, 2)).tolist()
        }
        return results

def main():
    path = "/scratch/s194260/BENDR/bendr_data/preprocess_mmidb_ica_combined"
    data_files = glob(f"{path}/*.hdf5")
    
    # Use a process pool to run tasks in parallel
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(process_file, data_files), total=len(data_files)))
    
    # Combine results into a single dictionary
    combined_results = {key: [result for item in results for result in item[key]] for key in results[0]}
    
    # Create a pandas DataFrame 
    df = pd.DataFrame(combined_results)
    
    # Save the DataFrame to a CSV file
    df.to_csv("/home/agjma/EEGatScale/mmidb_ica_data_statistics.csv", index=False)

if __name__ == "__main__":
    main()