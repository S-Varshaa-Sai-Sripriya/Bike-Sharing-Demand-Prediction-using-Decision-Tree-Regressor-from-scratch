import numpy as np
from src.utils.logger import logging
from src.utils.exceptions import CustomException

class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor implementation from scratch.
    Splits data based on variance reduction (MSE-based).
    """

    def __init__(self, min_samples_split: int = 2, max_depth: int = 5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Decision Tree Regressor.
        """
        try:
            logging.info("Training Decision Tree Regressor from scratch")
            self.tree = self._build_tree(X, y, depth=0)
        except Exception as e:
            raise CustomException(e)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input samples.
        """
        try:
            return np.array([self._predict_sample(x, self.tree) for x in X])
        except Exception as e:
            raise CustomException(e)

    def _mse(self, y: np.ndarray) -> float:
        """
        Calculate Mean Squared Error for target values.
        """
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the best split for the dataset using variance reduction.
        """
        best_feature, best_threshold, best_score = None, None, float("inf")
        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold

                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue

                left_y, right_y = y[left_mask], y[right_mask]
                score = (len(left_y) * self._mse(left_y) + len(right_y) * self._mse(right_y)) / n_samples

                if score < best_score:
                    best_feature, best_threshold, best_score = feature_idx, threshold, score

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        """
        Recursively build the tree.
        """
        n_samples, n_features = X.shape

        # stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)

        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            return np.mean(y)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _predict_sample(self, x: np.ndarray, tree):
        """
        Predict value for a single sample.
        """
        if not isinstance(tree, dict):
            return tree

        feature = tree["feature"]
        threshold = tree["threshold"]

        if x[feature] <= threshold:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])
