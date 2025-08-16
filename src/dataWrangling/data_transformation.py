import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def transform_data(test_size: float = 0.2, random_state: int = 42):
    data_dir = "data"
    features_path = os.path.join(data_dir, "preprocessed_features.csv")
    targets_path = os.path.join(data_dir, "preprocessed_targets.csv")

    # Load preprocessed data
    X = pd.read_csv(features_path)
    y = pd.read_csv(targets_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Normalize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Save transformed data
    X_train.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)

    print("✅ Transformation completed. Train/test splits saved in data/.")

if __name__ == "__main__":
    transform_data()
