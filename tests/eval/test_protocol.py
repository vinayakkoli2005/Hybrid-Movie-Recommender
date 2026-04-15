import numpy as np
import pandas as pd
import pytest
from cf_pipeline.eval.protocol import build_candidate_matrix, eval_pipeline
from cf_pipeline.models.base import BaseRanker


# ── stub rankers (reused in test_run_experiment.py too) ──────────────────────

class _AlwaysFavorPositive(BaseRanker):
    """Always gives column-0 (positive) the highest score."""
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        s = np.zeros((len(user_ids), item_ids.shape[1]))
        s[:, 0] = 1.0
        return s


class _AlwaysFavorNegative(BaseRanker):
    """Always gives the positive the lowest score (rank = n_candidates)."""
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        s = np.ones((len(user_ids), item_ids.shape[1]))
        s[:, 0] = 0.0
        return s


class _RandomRanker(BaseRanker):
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        return self.rng.random(item_ids.shape)


# ── build_candidate_matrix ────────────────────────────────────────────────────

def test_candidate_matrix_shape():
    df = pd.DataFrame({
        "user_id":   [1, 2],
        "positive":  [10, 20],
        "negatives": [[11, 12, 13], [21, 22, 23]],
    })
    users, items = build_candidate_matrix(df)
    assert users.shape == (2,)
    assert items.shape == (2, 4)   # 1 pos + 3 neg


def test_candidate_matrix_positive_is_col0():
    df = pd.DataFrame({
        "user_id":   [1],
        "positive":  [99],
        "negatives": [[1, 2, 3]],
    })
    _, items = build_candidate_matrix(df)
    assert items[0, 0] == 99


# ── plan tests ────────────────────────────────────────────────────────────────

def test_perfect_ranker_gets_perfect_metrics():
    eval_set = pd.DataFrame({
        "user_id":   [1, 2],
        "positive":  [10, 20],
        "negatives": [[11, 12, 13], [21, 22, 23]],
    })
    metrics = eval_pipeline(_AlwaysFavorPositive(), eval_set, ks=(1,))
    assert metrics["HR@1"]   == 1.0
    assert metrics["NDCG@1"] == 1.0


def test_random_ranker_is_not_perfect():
    eval_set = pd.DataFrame({
        "user_id":   list(range(1, 11)),
        "positive":  [1] * 10,
        "negatives": [[2, 3, 4, 5]] * 10,
    })
    metrics = eval_pipeline(_RandomRanker(seed=0), eval_set, ks=(1,))
    assert metrics["HR@1"] < 1.0


# ── extra correctness ─────────────────────────────────────────────────────────

def test_worst_ranker_gets_zero_at_k_less_than_n():
    eval_set = pd.DataFrame({
        "user_id":   [1, 2, 3],
        "positive":  [10, 20, 30],
        "negatives": [[11, 12, 13, 14]] * 3,   # 4 neg → 5 candidates, positive always last
    })
    metrics = eval_pipeline(_AlwaysFavorNegative(), eval_set, ks=(1, 4, 5))
    assert metrics["HR@1"]   == 0.0
    assert metrics["NDCG@1"] == 0.0
    assert metrics["HR@4"]   == 0.0   # rank=5, k=4 → still miss
    assert metrics["HR@5"]   == 1.0   # k=5 = n_candidates → everyone hits


def test_eval_pipeline_returns_all_k_keys():
    eval_set = pd.DataFrame({
        "user_id":   [1],
        "positive":  [1],
        "negatives": [list(range(2, 102))],   # 100 neg → 101 candidates
    })
    metrics = eval_pipeline(_AlwaysFavorPositive(), eval_set, ks=(1, 5, 10, 20))
    assert set(metrics.keys()) == {"HR@1", "HR@5", "HR@10", "HR@20",
                                    "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@20"}


def test_score_shape_mismatch_raises():
    class _BadRanker(BaseRanker):
        def score(self, user_ids, item_ids):
            return np.ones((len(user_ids), item_ids.shape[1] + 1))   # wrong shape

    eval_set = pd.DataFrame({
        "user_id":   [1],
        "positive":  [10],
        "negatives": [[11, 12]],
    })
    with pytest.raises(AssertionError):
        eval_pipeline(_BadRanker(), eval_set, ks=(1,))


def test_base_ranker_raises_not_implemented():
    ranker = BaseRanker()
    with pytest.raises(NotImplementedError):
        ranker.score(np.array([1]), np.array([[1, 2, 3]]))
