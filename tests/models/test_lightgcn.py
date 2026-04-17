from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cf_pipeline.models.lightgcn import LightGCNRanker


def test_lightgcn_one_epoch_smoke():
    torch.manual_seed(0)
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3],
        "item_id": [10, 20, 20, 30, 10, 30],
    })

    model = LightGCNRanker(
        emb_dim=8,
        n_layers=2,
        n_epochs=2,
        batch_size=4,
        lr=0.05,
    ).fit(train)
    scores = model.score(np.array([1, 2]), np.array([[20, 30], [10, 30]]))

    assert scores.shape == (2, 2)
