import os
import pandas as pd
from src.model.model import DecisionTreeRegressorScratch
from src.utils.utils import load_csv, save_object
from src.utils.logger import logger

def train(X_train_path: str, y_train_path: str):
    # Load training data
    X_train = load_csv(X_train_path).values
    y_train = load_csv(y_train_path).values.ravel()  # ensure 1D

    # Initialize model
    model = DecisionTreeRegressorScratch(max_depth=5, min_samples_split=2)

    # Fit model
    logger.info("Training started...")
    model.fit(X_train, y_train)
    logger.info("Training completed!")

    # Save model
    model_dir = "models"
    model_filename = "decision_tree.pkl"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    save_object(model_path, model)  # ✅ fixed argument order
    logger.info(f"Model saved successfully at {model_path}")

    return model

if __name__ == "__main__":
    train(
        X_train_path="data/X_train.csv",
        y_train_path="data/y_train.csv"
    )
