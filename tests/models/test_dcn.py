from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cf_pipeline.models.dcn import DCNRanker


def test_dcn_smoke():
    torch.manual_seed(0)
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [10, 20, 20, 30, 10],
    })

    model = DCNRanker(
        emb_dim=8,
        cross_layers=2,
        deep=(16, 8),
        dropout=0.1,
        n_epochs=2,
        batch_size=4,
    ).fit(train)
    scores, var = model.score_with_uncertainty(
        np.array([1]),
        np.array([[10, 99]]),
        n_mc=3,
    )

    assert scores.shape == (1, 2)
    assert var.shape == (1, 2)
