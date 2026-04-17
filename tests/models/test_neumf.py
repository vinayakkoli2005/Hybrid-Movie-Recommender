from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cf_pipeline.models.neumf import NeuMFRanker


def test_neumf_smoke():
    torch.manual_seed(0)
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [10, 20, 10, 30, 40],
    })

    model = NeuMFRanker(
        emb_dim=8,
        mlp_layers=(16, 8),
        n_epochs=3,
        batch_size=4,
    ).fit(train)
    scores = model.score(np.array([1]), np.array([[10, 30]]))

    assert scores.shape == (1, 2)
