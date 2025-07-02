#!/usr/bin/env python3
"""
Download script for Triple Pendulum experimental data.

This script downloads the experimental triple pendulum data used in the SREINet paper.
The data is hosted on a public repository and will be downloaded to the appropriate directory.
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

def main():
    """Main download function."""
    # Create data directory if it doesn't exist
    data_dir = Path("TriplePendulum_Data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Data URLs from the Multi-Arm Pendulum repository
    base_url = "https://raw.githubusercontent.com/dynamicslab/MultiArm-Pendulum/main/Datas/TriplePendulum"
    data_files = {
        "TripleDataFreeSwing_1_Dt_0_0001.mat": f"{base_url}/TripleDataFreeSwing_1_Dt_0_0001.mat",
        "TripleDataFreeSwing_2_Dt_0_0001.mat": f"{base_url}/TripleDataFreeSwing_2_Dt_0_0001.mat", 
        "TripleDataFreeSwing_3_Dt_0_0001.mat": f"{base_url}/TripleDataFreeSwing_3_Dt_0_0001.mat"
    }
    
    print("Triple Pendulum Data Downloader")
    print("=" * 40)
    print("This script will download the experimental triple pendulum data.")
    print("Total size: ~65MB")
    print()
    
    # Check if files already exist
    existing_files = []
    for filename in data_files.keys():
        if (data_dir / filename).exists():
            existing_files.append(filename)
    
    if existing_files:
        print(f"Found existing files: {', '.join(existing_files)}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    try:
        for filename, url in data_files.items():
            filepath = data_dir / filename
            if not filepath.exists():
                download_file(url, filepath)
            else:
                print(f"{filename} already exists, skipping...")
        
        print("\nDownload completed successfully!")
        print(f"Data files saved to: {data_dir.absolute()}")
        print("\nPlease see the DataExplanation.md file for details about the data format.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        print("\nAlternative download methods:")
        print("1. Manual download from: [URL to be provided]")
        print("2. Contact the authors for data access")
        sys.exit(1)

if __name__ == "__main__":
    main() 