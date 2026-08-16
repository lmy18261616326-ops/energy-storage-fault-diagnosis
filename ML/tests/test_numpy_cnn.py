from __future__ import annotations

import numpy as np

from energy_fault_ml.numpy_cnn import NumpyConv1DClassifier


def test_numpy_cnn_learns_a_simple_sequence_shift() -> None:
    rng = np.random.default_rng(7)
    negative = rng.normal(-1.0, 0.35, size=(40, 40))
    positive = rng.normal(1.0, 0.35, size=(40, 40))
    values = np.vstack([negative, positive])
    labels = np.repeat([0, 1], 40)
    model = NumpyConv1DClassifier(
        n_filters=6,
        kernel_size=5,
        stride=3,
        max_epochs=60,
        batch_size=32,
        n_iter_no_change=10,
        random_state=11,
    )
    model.fit(values, labels, sample_weight=np.ones(len(labels)))
    probability = model.predict_proba(values)
    assert probability.shape == (80, 2)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-10)
    assert np.mean(model.predict(values) == labels) >= 0.95
