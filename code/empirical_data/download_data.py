#!/usr/bin/env python3
"""
Download script for Triple and Double Pendulum experimental data.

This script downloads the experimental triple and double pendulum data used in the SREINet paper.
The data is hosted on a public repository and will be downloaded to the appropriate directories.
"""

import os
import requests
import zipfile
from pathlib import Path
import sys

def download_file(url, filename):
    """Download a file with progress bar."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print()

def download_pendulum_data(pendulum_type, data_files, data_dir):
    """Download pendulum data for a specific type (triple or double)."""
    print(f"{pendulum_type} Pendulum Data Downloader")
    print("=" * 40)
    print(f"This script will download the experimental {pendulum_type.lower()} pendulum data.")
    
    # Calculate total size (approximate sizes in MB)
    if pendulum_type == "Double":
        total_size_mb = 4.7 + 5.0 + 4.2 + 4.6 + 22.6 + 24.9  # All double pendulum files
    else:
        total_size_mb = 65  # Triple pendulum files
    print(f"Total size: ~{total_size_mb}MB")
    print()
    
    # Check if files already exist
    existing_files = []
    for filename in data_files.keys():
        if (data_dir / filename).exists():
            existing_files.append(filename)
    
    if existing_files:
        print(f"Found existing files: {', '.join(existing_files)}")
        print("Skipping existing files, will only download missing files.")
    
    try:
        for filename, url in data_files.items():
            filepath = data_dir / filename
            if not filepath.exists():
                download_file(url, filepath)
            else:
                print(f"{filename} already exists, skipping...")
        
        print(f"\n{pendulum_type} pendulum download completed successfully!")
        print(f"Data files saved to: {data_dir.absolute()}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {pendulum_type.lower()} pendulum data: {e}")
        return False

def main():
    """Main download function."""
    # Create data directories if they don't exist
    triple_data_dir = Path("TriplePendulum_Data")
    double_data_dir = Path("DoublePendulum_Data")
    triple_data_dir.mkdir(parents=True, exist_ok=True)
    double_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Triple pendulum data URLs
    triple_base_url = "https://raw.githubusercontent.com/dynamicslab/MultiArm-Pendulum/main/Datas/TriplePendulum"
    triple_data_files = {
        "TripleDataFreeSwing_1_Dt_0_0001.mat": f"{triple_base_url}/TripleDataFreeSwing_1_Dt_0_0001.mat",
        "TripleDataFreeSwing_2_Dt_0_0001.mat": f"{triple_base_url}/TripleDataFreeSwing_2_Dt_0_0001.mat", 
        "TripleDataFreeSwing_3_Dt_0_0001.mat": f"{triple_base_url}/TripleDataFreeSwing_3_Dt_0_0001.mat"
    }
    
    # Double pendulum data URLs
    double_base_url = "https://raw.githubusercontent.com/dynamicslab/MultiArm-Pendulum/main/Datas/DoublePendulum"
    double_data_files = {
        "DoubleDataFreeSwing_1_Dt_0_001.mat": f"{double_base_url}/DoubleDataFreeSwing_1_Dt_0_001.mat",
        "DoubleDataFreeSwing_2_Dt_0_001.mat": f"{double_base_url}/DoubleDataFreeSwing_2_Dt_0_001.mat",
        "DoubleDataFreeSwing_3_Dt_0_001.mat": f"{double_base_url}/DoubleDataFreeSwing_3_Dt_0_001.mat",
        "DoubleDataFreeSwing_4_Dt_0_001.mat": f"{double_base_url}/DoubleDataFreeSwing_4_Dt_0_001.mat",
    }
    
    print("Pendulum Data Downloader")
    print("=" * 50)
    print("This script will download both triple and double pendulum experimental data.")
    print("Total size: ~166MB")
    print()
    
    success = True
    
    # Download both triple and double pendulum data
    success &= download_pendulum_data("Triple", triple_data_files, triple_data_dir)
    success &= download_pendulum_data("Double", double_data_files, double_data_dir)
    
    if success:
        print("\nAll downloads completed successfully!")
        print("\nPlease see the DataExplanation.md files for details about the data format.")
    else:
        print("\nSome downloads failed. Please check the error messages above.")
        print("\nAlternative download methods:")
        print("1. Manual download from: https://github.com/dynamicslab/MultiArm-Pendulum")
        print("2. Contact the authors for data access")
        sys.exit(1)

if __name__ == "__main__":
    main() 