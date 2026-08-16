"""Small dependency-free 1D CNN for tabular sequence comparison experiments."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class NumpyConv1DClassifier(ClassifierMixin, BaseEstimator):
    """One learned convolution, ReLU, global pooling, and softmax output.

    This estimator is intentionally compact.  It lets the research pipeline
    compare the inductive bias of a 1D convolution without adding a large deep
    learning runtime.  Input columns retain the event-feature ordering.
    """

    def __init__(
        self,
        *,
        n_filters: int = 12,
        kernel_size: int = 7,
        stride: int = 4,
        pool_segments: int = 8,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-4,
        max_epochs: int = 140,
        batch_size: int = 64,
        tol: float = 1e-5,
        n_iter_no_change: int = 15,
        random_state: int | None = None,
    ) -> None:
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_segments = pool_segments
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _patches(self, values: np.ndarray) -> np.ndarray:
        patches = np.lib.stride_tricks.sliding_window_view(
            values, self.kernel_size, axis=1
        )
        return patches[:, :: self.stride, :]

    def _forward(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        patches = self._patches(values)
        convolution = (
            np.einsum("nok,fk->nof", patches, self.conv_kernel_, optimize=True)
            + self.conv_bias_
        )
        activation = np.maximum(convolution, 0.0)
        pooled = np.concatenate(
            [
                activation[:, start:end, :].mean(axis=1)
                for start, end in self._pool_bounds(activation.shape[1])
            ],
            axis=1,
        )
        probability = self._softmax(pooled @ self.output_weight_ + self.output_bias_)
        return patches, convolution, pooled, probability

    def _pool_bounds(self, sequence_length: int) -> list[tuple[int, int]]:
        segments = min(self.pool_segments, sequence_length)
        edges = np.linspace(0, sequence_length, segments + 1, dtype=int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(segments)]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "NumpyConv1DClassifier":
        values, labels = check_X_y(X, y, dtype=np.float64)
        if self.kernel_size < 1 or self.kernel_size > values.shape[1]:
            raise ValueError("kernel_size must be within the input feature length")
        if self.stride < 1 or self.n_filters < 1 or self.pool_segments < 1:
            raise ValueError("stride, n_filters, and pool_segments must be positive")
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required")
        self.n_features_in_ = values.shape[1]
        weights = (
            np.ones(len(values), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if weights.shape != (len(values),):
            raise ValueError("sample_weight must contain one value per row")
        weights = weights / weights.mean()

        rng = np.random.default_rng(self.random_state)
        self.conv_kernel_ = rng.normal(
            0.0,
            np.sqrt(2.0 / self.kernel_size),
            size=(self.n_filters, self.kernel_size),
        )
        self.conv_bias_ = np.zeros(self.n_filters)
        output_length = 1 + (values.shape[1] - self.kernel_size) // self.stride
        pooled_width = self.n_filters * min(self.pool_segments, output_length)
        self.output_weight_ = rng.normal(
            0.0,
            np.sqrt(2.0 / pooled_width),
            size=(pooled_width, len(self.classes_)),
        )
        self.output_bias_ = np.zeros(len(self.classes_))

        parameters = (
            "conv_kernel_",
            "conv_bias_",
            "output_weight_",
            "output_bias_",
        )
        first_moment = {name: np.zeros_like(getattr(self, name)) for name in parameters}
        second_moment = {name: np.zeros_like(getattr(self, name)) for name in parameters}
        step = 0
        best_loss = np.inf
        stale_epochs = 0

        for epoch in range(self.max_epochs):
            order = rng.permutation(len(values))
            epoch_loss = 0.0
            epoch_weight = 0.0
            for start in range(0, len(values), self.batch_size):
                rows = order[start : start + self.batch_size]
                batch = values[rows]
                batch_labels = encoded[rows]
                batch_weights = weights[rows]
                patches, convolution, pooled, probability = self._forward(batch)
                safe = np.clip(probability[np.arange(len(rows)), batch_labels], 1e-12, 1.0)
                epoch_loss += float(np.sum(-np.log(safe) * batch_weights))
                epoch_weight += float(batch_weights.sum())

                gradient_logits = probability.copy()
                gradient_logits[np.arange(len(rows)), batch_labels] -= 1.0
                gradient_logits *= batch_weights[:, None] / batch_weights.sum()
                gradients: dict[str, np.ndarray] = {}
                gradients["output_weight_"] = (
                    pooled.T @ gradient_logits
                    + self.weight_decay * self.output_weight_
                )
                gradients["output_bias_"] = gradient_logits.sum(axis=0)
                gradient_pooled = gradient_logits @ self.output_weight_.T
                bounds = self._pool_bounds(convolution.shape[1])
                gradient_segments = gradient_pooled.reshape(
                    len(rows), len(bounds), self.n_filters
                )
                gradient_activation = np.zeros_like(convolution)
                for segment, (left, right) in enumerate(bounds):
                    gradient_activation[:, left:right, :] = (
                        gradient_segments[:, segment, None, :] / (right - left)
                    )
                gradient_convolution = gradient_activation * (convolution > 0.0)
                gradients["conv_kernel_"] = (
                    np.einsum(
                        "nof,nok->fk", gradient_convolution, patches, optimize=True
                    )
                    + self.weight_decay * self.conv_kernel_
                )
                gradients["conv_bias_"] = gradient_convolution.sum(axis=(0, 1))

                step += 1
                for name in parameters:
                    first_moment[name] = (
                        0.9 * first_moment[name] + 0.1 * gradients[name]
                    )
                    second_moment[name] = (
                        0.999 * second_moment[name] + 0.001 * gradients[name] ** 2
                    )
                    corrected_first = first_moment[name] / (1.0 - 0.9**step)
                    corrected_second = second_moment[name] / (1.0 - 0.999**step)
                    update = self.learning_rate * corrected_first / (
                        np.sqrt(corrected_second) + 1e-8
                    )
                    setattr(self, name, getattr(self, name) - update)

            mean_loss = epoch_loss / max(epoch_weight, 1e-12)
            if mean_loss < best_loss - self.tol:
                best_loss = mean_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.n_iter_no_change:
                    self.n_iter_ = epoch + 1
                    break
        else:
            self.n_iter_ = self.max_epochs
        self.loss_ = float(best_loss)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("conv_kernel_", "output_weight_", "classes_"))
        values = check_array(X, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count differs from training")
        return self._forward(values)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probability = self.predict_proba(X)
        return self.classes_[np.argmax(probability, axis=1)]
