"""
knn.py
------
K-Nearest Neighbors implementation from scratch using NumPy.
Responsible: Yuxiang Jiang (yj548)

Features:
  - Euclidean distance computation (vectorized)
  - Uniform and distance-weighted voting
  - Probability estimates (fraction of fraud neighbors)
  - Hyperparameter tuning via stratified k-fold cross-validation
  - Model persistence (save/load)

Usage:
  from knn import KNNClassifier
  model = KNNClassifier(k=5, weights='distance')
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  probabilities = model.predict_proba(X_test)
"""

import numpy as np
import os

from evaluation import (
    compute_all_metrics,
)

# ─────────────────────────────────────────────
# 1. KNN CLASSIFIER
# ─────────────────────────────────────────────


class KNNClassifier:
    """
    K-Nearest Neighbors classifier implemented from scratch with NumPy.

    Parameters
    ----------
    k : int
        Number of neighbors to consider.
    weights : str, {'uniform', 'distance'}
        'uniform'  — all k neighbors vote equally.
        'distance' — closer neighbors have higher weight (1/d).
    """

    def __init__(self, k: int = 5, weights: str = "uniform"):
        assert k >= 1, "k must be at least 1."
        assert weights in (
            "uniform",
            "distance",
        ), "weights must be 'uniform' or 'distance'."
        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Store the training data. KNN is a lazy learner — no actual
        training computation happens here.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)

        Returns
        -------
        self
        """
        self.X_train = X.astype(np.float64)
        self.y_train = y.astype(np.int64)
        self.classes_ = np.unique(y)
        return self

    # ── Distance Computation ──────────────────

    def _compute_distances(self, X_query: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between each query point and all
        training points using the vectorized expansion:

            ||a - b||^2 = ||a||^2 - 2*a·b + ||b||^2

        Parameters
        ----------
        X_query : np.ndarray, shape (M, D)

        Returns
        -------
        distances : np.ndarray, shape (M, N)
            distances[i, j] = Euclidean distance from query i to train j.
        """
        # ||query||^2 : shape (M, 1)
        query_sq = np.sum(X_query**2, axis=1, keepdims=True)
        # ||train||^2 : shape (1, N)
        train_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T
        # cross term : shape (M, N)
        cross = X_query @ self.X_train.T

        # squared distances
        dist_sq = query_sq - 2 * cross + train_sq
        # Clamp numerical errors (tiny negatives) to zero
        dist_sq = np.maximum(dist_sq, 0.0)

        return np.sqrt(dist_sq)

    # ── Voting ────────────────────────────────

    def _vote(self, neighbor_labels: np.ndarray, neighbor_dists: np.ndarray) -> tuple:
        """
        Perform voting for a SINGLE query point.

        Parameters
        ----------
        neighbor_labels : np.ndarray, shape (k,)
        neighbor_dists  : np.ndarray, shape (k,)

        Returns
        -------
        predicted_class : int
        fraud_probability : float
            Weighted proportion of fraud (class=1) among neighbors.
        """
        if self.weights == "uniform":
            weights = np.ones(self.k)
        else:
            # distance weighting: w_i = 1 / (d_i + epsilon)
            # epsilon prevents division by zero for exact matches
            weights = 1.0 / (neighbor_dists + 1e-8)

        # Weighted vote for each class
        vote_0 = np.sum(weights[neighbor_labels == 0])
        vote_1 = np.sum(weights[neighbor_labels == 1])

        total = vote_0 + vote_1
        fraud_prob = vote_1 / total if total > 0 else 0.0
        predicted = 1 if vote_1 > vote_0 else 0

        return predicted, fraud_prob

    # ── Prediction ────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the query set.

        Parameters
        ----------
        X : np.ndarray, shape (M, D)

        Returns
        -------
        y_pred : np.ndarray, shape (M,)
        """
        _, y_pred, _ = self._predict_internal(X)
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the query set.

        Parameters
        ----------
        X : np.ndarray, shape (M, D)

        Returns
        -------
        proba : np.ndarray, shape (M, 2)
            Column 0 = P(class=0), Column 1 = P(class=1).
        """
        _, _, proba = self._predict_internal(X)
        return proba

    def _predict_internal(self, X: np.ndarray) -> tuple:
        """
        Core prediction logic. Computes distances, finds k nearest
        neighbors, and performs voting.

        To handle large test sets efficiently, we process in batches
        to avoid creating huge (M x N) distance matrices.

        Returns
        -------
        distances_all : None (not stored for memory)
        y_pred : np.ndarray, shape (M,)
        proba  : np.ndarray, shape (M, 2)
        """
        assert self.X_train is not None, "Model not fitted. Call fit() first."

        M = X.shape[0]
        y_pred = np.empty(M, dtype=np.int64)
        proba = np.empty((M, 2), dtype=np.float64)

        # Process in batches to manage memory
        batch_size = 1000
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            X_batch = X[start:end]

            # Compute distances: shape (batch, N_train)
            dists = self._compute_distances(X_batch)

            # Find k nearest neighbors for each query in batch
            # argpartition is O(N) vs O(N log N) for full sort
            knn_indices = np.argpartition(dists, self.k, axis=1)[:, : self.k]

            for i in range(end - start):
                idx = knn_indices[i]
                neighbor_labels = self.y_train[idx]
                neighbor_dists = dists[i, idx]

                pred, fraud_prob = self._vote(neighbor_labels, neighbor_dists)
                y_pred[start + i] = pred
                proba[start + i, 1] = fraud_prob
                proba[start + i, 0] = 1.0 - fraud_prob

        return None, y_pred, proba

    # ── Model Persistence ─────────────────────

    def save(self, save_dir: str = "saved_models") -> None:
        """
        Save model state to disk.
        Saves: X_train, y_train, hyperparameters (k, weights).
        """
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "knn_X_train.npy"), self.X_train)
        np.save(os.path.join(save_dir, "knn_y_train.npy"), self.y_train)
        np.save(
            os.path.join(save_dir, "knn_params.npy"),
            np.array([self.k, self.weights], dtype=object),
        )
        print(f"  KNN model saved to {save_dir}/")

    @classmethod
    def load(cls, save_dir: str = "saved_models") -> "KNNClassifier":
        """
        Load a saved KNN model from disk.
        """
        params = np.load(os.path.join(save_dir, "knn_params.npy"), allow_pickle=True)
        k = int(params[0])
        weights = str(params[1])

        model = cls(k=k, weights=weights)
        model.X_train = np.load(os.path.join(save_dir, "knn_X_train.npy"))
        model.y_train = np.load(os.path.join(save_dir, "knn_y_train.npy"))
        model.classes_ = np.unique(model.y_train)
        print(f"  KNN model loaded from {save_dir}/ (k={k}, weights={weights})")
        return model


# ─────────────────────────────────────────────
# 2. STRATIFIED K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────


def stratified_kfold_split(
    y: np.ndarray, n_folds: int = 5, random_state: int = 42
) -> list:
    """
    Generate stratified k-fold indices.

    Parameters
    ----------
    y : np.ndarray, shape (N,) — labels
    n_folds : int
    random_state : int

    Returns
    -------
    folds : list of (train_indices, val_indices) tuples
    """
    rng = np.random.default_rng(random_state)

    # Separate indices by class
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    # Split each class into n_folds roughly equal parts
    folds_0 = np.array_split(idx_0, n_folds)
    folds_1 = np.array_split(idx_1, n_folds)

    folds = []
    for i in range(n_folds):
        val_idx = np.concatenate([folds_0[i], folds_1[i]])
        train_idx = np.concatenate(
            [folds_0[j] for j in range(n_folds) if j != i]
            + [folds_1[j] for j in range(n_folds) if j != i]
        )
        folds.append((train_idx, val_idx))

    return folds


# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list = None,
    weight_options: list = None,
    n_folds: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Grid search over k and weighting scheme using stratified k-fold CV.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, D)
    y_train : np.ndarray, shape (N,)
    k_values : list of int — values of k to try
    weight_options : list of str — ['uniform', 'distance']
    n_folds : int — number of CV folds
    scoring : str — metric to optimize ('precision', 'recall', 'f1', 'auc_roc')
    random_state : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'best_k', 'best_weights', 'best_score',
        'all_results' (list of dicts for each configuration)
    """
    if k_values is None:
        k_values = [3, 5, 7, 11, 15]
    if weight_options is None:
        weight_options = ["uniform", "distance"]

    folds = stratified_kfold_split(y_train, n_folds=n_folds, random_state=random_state)

    all_results = []
    best_score = -1.0
    best_k = None
    best_weights = None

    total_configs = len(k_values) * len(weight_options)
    config_num = 0

    if verbose:
        print(f"\n{'=' * 60}")
        print("  KNN Hyperparameter Tuning")
        print(f"  k values: {k_values}")
        print(f"  weights:  {weight_options}")
        print(f"  {n_folds}-fold CV  |  scoring: {scoring}")
        print(f"{'=' * 60}")

    for w in weight_options:
        for k in k_values:
            config_num += 1
            fold_scores = []

            for fold_i, (train_idx, val_idx) in enumerate(folds):
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]

                model = KNNClassifier(k=k, weights=w)
                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                metrics = compute_all_metrics(y_val, y_pred, y_proba)
                fold_scores.append(metrics[scoring])

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            result = {
                "k": k,
                "weights": w,
                "mean_score": mean_score,
                "std_score": std_score,
                "fold_scores": fold_scores,
            }
            all_results.append(result)

            if verbose:
                print(
                    f"  [{config_num:2d}/{total_configs}]  "
                    f"k={k:2d}  weights={w:8s}  "
                    f"{scoring}={mean_score:.4f} (+/- {std_score:.4f})"
                )

            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_weights = w

    if verbose:
        print(
            f"\n  >>> Best: k={best_k}, weights={best_weights}, "
            f"{scoring}={best_score:.4f}"
        )
        print(f"{'=' * 60}\n")

    return {
        "best_k": best_k,
        "best_weights": best_weights,
        "best_score": best_score,
        "all_results": all_results,
    }
