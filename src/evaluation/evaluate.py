import numpy as np
from src.model.model import DecisionTreeRegressorScratch
from src.utils.utils import load_object, load_csv
from src.evaluation.metrics import mean_squared_error, r2_score

def evaluate(model_path: str, X_test_path: str, y_test_path: str):
    # Load model and data
    model = load_object(model_path)
    X_test = load_csv(X_test_path).values   # convert DataFrame → numpy
    y_test = load_csv(y_test_path).values.flatten()

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f" Model Evaluation Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²:  {r2:.4f}")

    return {"mse": mse, "r2": r2}

if __name__ == "__main__":
    evaluate(
        model_path="models/decision_tree.pkl",
        X_test_path="data/X_test.csv",
        y_test_path="data/y_test.csv"
    )
