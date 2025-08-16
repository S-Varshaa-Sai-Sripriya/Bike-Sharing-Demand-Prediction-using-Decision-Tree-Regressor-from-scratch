import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    Compute R² Score (coefficient of determination).
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
