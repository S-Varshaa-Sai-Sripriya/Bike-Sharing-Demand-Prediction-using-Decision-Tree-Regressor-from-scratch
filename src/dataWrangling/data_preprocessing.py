import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    data_dir = "data"
    features_path = os.path.join(data_dir, "features.csv")
    targets_path = os.path.join(data_dir, "targets.csv")

    # Load features and targets
    X = pd.read_csv(features_path)
    y = pd.read_csv(targets_path)

    # Handle missing values (simple fill for now)
    X = X.fillna(method="ffill")

    # Encode categorical columns if any
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Save preprocessed features and targets
    X.to_csv(os.path.join(data_dir, "preprocessed_features.csv"), index=False)
    y.to_csv(os.path.join(data_dir, "preprocessed_targets.csv"), index=False)

    print("✅ Preprocessing completed. Saved preprocessed_features.csv and preprocessed_targets.csv in data/.")

if __name__ == "__main__":
    preprocess_data()
