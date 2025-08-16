import pandas as pd
import numpy as np
import os
import pickle
from src.utils.logger import logger
from src.utils.exceptions import CustomException

def save_csv(data: pd.DataFrame, filepath: str) -> None:
    """
    Saves a DataFrame to CSV.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        raise CustomException(f"Error saving CSV to {filepath}", e)

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Loads a DataFrame from CSV.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found!")
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise CustomException(f"Error loading CSV from {filepath}", e)

def save_numpy(file_path: str, array: np.ndarray) -> None:
    """Save numpy array to .npy file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, array)

def load_numpy(file_path: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(file_path, allow_pickle=True)

def save_object(file_path: str, obj) -> None:
    """Save Python object using pickle."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {file_path}")

def load_object(file_path: str):
    """Load Python object from pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Object loaded from {file_path}")
    return obj
