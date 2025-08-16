# src/dataWrangling/data_ingestion.py

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_uci_dataset(dataset_id: int, data_dir: str = "data") -> None:
    """
    Loads a dataset from the UCI repository using ucimlrepo and saves it locally as CSV.

    Args:
        dataset_id (int): UCI dataset ID (refer to https://archive.ics.uci.edu/ml/index.php).
        data_dir (str): Directory where CSV files will be saved.
    """
    # Fetch dataset
    dataset = fetch_ucirepo(id=dataset_id)

    # Extract features and targets
    X = dataset.data.features
    y = dataset.data.targets

    # Create data directory if not exists
    os.makedirs(data_dir, exist_ok=True)

    # Save as CSV
    X.to_csv(os.path.join(data_dir, "features.csv"), index=False)
    y.to_csv(os.path.join(data_dir, "targets.csv"), index=False)

    print(f"✅ Dataset '{dataset.metadata['name']}' (ID: {dataset_id}) loaded successfully.")
    print(f"📁 Features saved to {os.path.join(data_dir, 'features.csv')}")
    print(f"📁 Targets saved to {os.path.join(data_dir, 'targets.csv')}")

if __name__ == "__main__":
    # Example: Load Bike Sharing (ID: 275) — change the ID for another dataset
    load_uci_dataset(dataset_id=275)
