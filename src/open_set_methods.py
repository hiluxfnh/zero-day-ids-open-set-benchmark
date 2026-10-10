"""
open_set_methods.py
Candidate open-set / zero-day detectors, all deliberately built on
lightweight, CPU-friendly libraries (scikit-learn + numpy only -- no
PyTorch/TensorFlow) so training and inference cost are representative of
what a resource-constrained IoT gateway could realistically run. This is a
methodological choice, not a limitation: the paper's contribution is a
benchmark of accuracy-vs-cost trade-offs, so every method must be
measurable on the same lightweight footing.

Each detector implements:
  .fit(X_benign)                 -- trained on BENIGN traffic only (+ known
                                     attacks, where noted) to score novelty
  .score(X)                      -- higher score = more anomalous
and is wrapped so run_experiment.py can profile fit/predict wall-clock time
and serialized model size uniformly.
"""
from __future__ import annotations
import time
import pickle
import io
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.covariance import LedoitWolf


class BaseDetector:
    name = "base"

    def fit(self, X_benign: np.ndarray, X_known_attacks: np.ndarray | None = None):
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores, higher = more anomalous."""
        raise NotImplementedError

    def model_size_bytes(self) -> int:
        buf = io.BytesIO()
        pickle.dump(self, buf)
        return buf.tell()


class IsolationForestDetector(BaseDetector):
    name = "IsolationForest"

    def __init__(self, n_estimators=100, random_state=42):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def fit(self, X_benign, X_known_attacks=None):
        self.model.fit(X_benign)
        return self

    def score(self, X):
        # decision_function: higher = more normal, so invert
        return -self.model.decision_function(X)


class OneClassSVMDetector(BaseDetector):
    name = "OneClassSVM"

    def __init__(self, nu=0.05, kernel="rbf", gamma="scale", max_train=4000):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.max_train = max_train  # OC-SVM is O(n^2)-ish; cap training set for laptop feasibility

    def fit(self, X_benign, X_known_attacks=None):
        if len(X_benign) > self.max_train:
            idx = np.random.RandomState(42).choice(len(X_benign), self.max_train, replace=False)
            X_benign = X_benign[idx]
        self.model.fit(X_benign)
        return self

    def score(self, X):
        return -self.model.decision_function(X)


class PCAReconstructionDetector(BaseDetector):
    """Reconstruction-error anomaly detector: project to a low-rank PCA
    subspace fit on benign traffic, score = reconstruction error. This is
    the classic linear stand-in for an autoencoder -- cheap and a fair
    lightweight baseline."""
    name = "PCA-Reconstruction"

    def __init__(self, n_components=0.95):
        self.model = PCA(n_components=n_components, random_state=42)

    def fit(self, X_benign, X_known_attacks=None):
        self.model.fit(X_benign)
        return self

    def score(self, X):
        X_proj = self.model.transform(X)
        X_rec = self.model.inverse_transform(X_proj)
        return np.mean((X - X_rec) ** 2, axis=1)


class MLPAutoencoderDetector(BaseDetector):
    """Shallow bottleneck MLP trained to reconstruct benign traffic
    (input -> bottleneck -> output). A lightweight, dependency-free
    (no torch) neural analogue to the deep-autoencoder baselines used in
    prior IDS work, sized so it trains in seconds on CPU."""
    name = "MLP-Autoencoder"

    def __init__(self, bottleneck=8, hidden=32, max_iter=200):
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.model = None
        self.max_iter = max_iter

    def fit(self, X_benign, X_known_attacks=None):
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.bottleneck, self.hidden),
            activation="relu", solver="adam", max_iter=self.max_iter,
            random_state=42, early_stopping=True,
        )
        self.model.fit(X_benign, X_benign)
        return self

    def score(self, X):
        X_rec = self.model.predict(X)
        return np.mean((X - X_rec) ** 2, axis=1)


class PrototypeContrastiveDetector(BaseDetector):
    """Lightweight stand-in for contrastive open-set methods (cf. CLOSR /
    CLAD, arXiv:2601.09902): learns a per-known-class prototype (centroid)
    in a whitened feature space fit on benign + known attacks, and scores
    novelty as the Mahalanobis distance to the nearest prototype. This
    captures the same "distance-to-known-classes" scoring principle as
    deep contrastive open-set recognition, without requiring a GPU/torch
    training loop -- appropriate given the resource-constrained framing of
    the study. Reported explicitly in the paper as a lightweight
    approximation, benchmarked against the literature's heavier
    contrastive-loss models on the accuracy side.
    """
    name = "Prototype-Contrastive"

    def __init__(self):
        # LedoitWolf: adaptive shrinkage covariance estimator. Plain
        # EmpiricalCovariance is numerically unstable here because several
        # CICIoT2023 features are near-constant binary protocol flags,
        # which makes the raw covariance matrix close to singular and
        # produces erratic Mahalanobis distances once inverted. LedoitWolf
        # regularizes toward a well-conditioned target, which is the
        # standard fix for exactly this failure mode.
        self.cov = LedoitWolf()
        self.prototypes = {}
        self.precision_ = None

    def fit(self, X_benign, X_known_attacks=None, y_known=None):
        # Fit a shared covariance (whitening) on all known-labeled data.
        X_all = X_benign if X_known_attacks is None else np.vstack([X_benign, X_known_attacks])
        self.cov.fit(X_all)
        self.precision_ = self.cov.precision_

        self.prototypes["Benign"] = X_benign.mean(axis=0)
        if X_known_attacks is not None and y_known is not None:
            for label in np.unique(y_known):
                self.prototypes[label] = X_known_attacks[y_known == label].mean(axis=0)
        return self

    def _mahalanobis(self, X, mean):
        diff = X - mean
        return np.einsum("ij,jk,ik->i", diff, self.precision_, diff)

    def score(self, X):
        dists = np.vstack([self._mahalanobis(X, mean) for mean in self.prototypes.values()])
        return dists.min(axis=0)  # distance to nearest known prototype; high = novel


def profile_detector(detector: BaseDetector, X_train, X_test, **fit_kwargs):
    """Fit + score a detector, returning scores plus cost metrics."""
    t0 = time.perf_counter()
    detector.fit(X_train, **fit_kwargs)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = detector.score(X_test)
    infer_time = time.perf_counter() - t0
    per_sample_ms = 1000 * infer_time / max(len(X_test), 1)

    return {
        "scores": scores,
        "train_time_s": train_time,
        "infer_time_total_s": infer_time,
        "infer_time_per_sample_ms": per_sample_ms,
        "model_size_kb": detector.model_size_bytes() / 1024,
    }


ALL_DETECTORS = {
    "IsolationForest": IsolationForestDetector,
    "OneClassSVM": OneClassSVMDetector,
    "PCA-Reconstruction": PCAReconstructionDetector,
    "MLP-Autoencoder": MLPAutoencoderDetector,
    "Prototype-Contrastive": PrototypeContrastiveDetector,
}
