"""Evaluation protocol for the NCF leave-one-out benchmark.

Every model is evaluated the same way:
  1. build_candidate_matrix  →  (user_ids, item_ids) arrays
  2. model.score(user_ids, item_ids)  →  score matrix
  3. all_metrics(scores)  →  dict of HR@K / NDCG@K values
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.models.base import BaseRanker
from cf_pipeline.utils.io import save_result


def build_candidate_matrix(
    eval_set: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an eval DataFrame into dense arrays for scoring.

    Args:
        eval_set: DataFrame with columns [user_id, positive, negatives].
                  'negatives' is a list of item IDs per row.

    Returns:
        user_ids: (n_users,) array of user IDs.
        item_ids: (n_users, 1 + n_neg) array of item IDs.
                  Column 0 is the positive; remaining columns are negatives.
    """
    user_ids = eval_set["user_id"].to_numpy()
    pos      = eval_set["positive"].to_numpy().reshape(-1, 1)        # (n, 1)
    neg      = np.stack([np.asarray(x) for x in eval_set["negatives"].tolist()])  # (n, n_neg)
    item_ids = np.concatenate([pos, neg], axis=1)                    # (n, 1+n_neg)
    return user_ids, item_ids


def eval_pipeline(
    model: BaseRanker,
    eval_set: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Score every (user, candidate-set) pair and compute ranking metrics.

    Args:
        model:    Any BaseRanker subclass.
        eval_set: DataFrame with columns [user_id, positive, negatives].
        ks:       K values for HR@K and NDCG@K.

    Returns:
        Dict of metric names → values, e.g. {'HR@1': 0.31, 'NDCG@10': 0.42, …}
    """
    user_ids, item_ids = build_candidate_matrix(eval_set)
    scores = model.score(user_ids, item_ids)
    assert scores.shape == item_ids.shape, (
        f"score() returned shape {scores.shape}, "
        f"expected {item_ids.shape}. "
        "Make sure your model returns one score per (user, candidate) pair."
    )
    return all_metrics(scores, ks=ks)


def run_and_save_experiment(
    model: BaseRanker,
    eval_set: pd.DataFrame,
    experiment_name: str,
    out_path: str | Path,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict:
    """Evaluate a model, print a summary, and persist results to disk.

    The saved JSON includes the experiment name, all metrics, and a timestamp
    + git SHA (added automatically by save_result).

    Args:
        model:           Any BaseRanker subclass.
        eval_set:        DataFrame with columns [user_id, positive, negatives].
        experiment_name: Human-readable label stored in the result file.
        out_path:        Where to write the JSON result.
        ks:              K values for HR@K and NDCG@K.

    Returns:
        The payload dict that was saved.
    """
    metrics = eval_pipeline(model, eval_set, ks=ks)
    payload = {"experiment": experiment_name, "metrics": metrics}
    save_result(payload, out_path)
    return payload
