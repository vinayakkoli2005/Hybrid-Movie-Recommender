from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cf_pipeline.models.bpr_mf import BPRMFRanker


def test_bpr_mf_trains_and_scores_personalized():
    torch.manual_seed(0)
    train = pd.DataFrame({
        "user_id": [1] * 10 + [2] * 10,
        "item_id": list(range(0, 20, 2)) + list(range(1, 20, 2)),
    })

    model = BPRMFRanker(emb_dim=8, n_epochs=20, lr=0.05, batch_size=4).fit(train)
    user_ids = np.array([1, 2])
    item_ids = np.array([[100, 101], [101, 100]])
    scores = model.score(user_ids, item_ids)

    assert scores.shape == (2, 2)
