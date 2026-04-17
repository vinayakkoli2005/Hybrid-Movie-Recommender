from __future__ import annotations

import numpy as np
import pandas as pd

from cf_pipeline.models.ease import EASERRanker


def test_ease_diagonal_zero_after_fit():
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 20, 30],
    })

    model = EASERRanker(reg_lambda=10.0).fit(train)

    assert model._B is not None
    assert np.allclose(np.diag(model._B), 0.0)


def test_ease_personalized_score():
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 20, 30],
    })

    model = EASERRanker(reg_lambda=10.0).fit(train)
    scores = model.score(np.array([1]), np.array([[30, 99]]))

    assert scores[0, 0] >= scores[0, 1]
