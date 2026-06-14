import os
import glob
import kagglehub
import pandas as pd

def load_cinema_data():
    """
    Downloads the latest version of the cinema ticket dataset from Kaggle
    and loads the primary data file into a Pandas DataFrame.
    """
    print("Downloading dataset from Kaggle Hub...")
    # Download latest version
    path = kagglehub.dataset_download("arashnic/cinema-ticket")
    print("Path to dataset files:", path)
    
    # Look for CSV files in the downloaded path
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the downloaded dataset path: {path}")
    
    # The dataset contains 'cinemaTicket_Ref.csv'. We will grab the first available CSV.
    df = pd.read_csv(csv_files[0])
    return df

if __name__ == "__main__":
    # Test script locally
    data = load_cinema_data()
    print(f"Dataset successfully loaded! Shape: {data.shape}")
    print(data.head())