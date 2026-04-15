"""Unit tests for baseline rankers (Task 15).

Tests are deliberately self-contained — no disk I/O, no fixtures from
prepare_data.py.  They verify the mathematical contract of each model class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cf_pipeline.models.baselines import PopularityRanker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_train() -> pd.DataFrame:
    """Minimal training DataFrame: items {1,2,3} with counts {3,2,1}."""
    return pd.DataFrame({
        "user_id": [0, 0, 1, 1, 2, 3],
        "item_id": [1, 2, 1, 3, 1, 2],   # item 1 × 3, item 2 × 2, item 3 × 1
        "timestamp": list(range(6)),
    })


@pytest.fixture()
def fitted_pop(tiny_train) -> PopularityRanker:
    return PopularityRanker().fit(tiny_train)


# ---------------------------------------------------------------------------
# PopularityRanker
# ---------------------------------------------------------------------------

class TestPopularityRankerFit:
    def test_fit_returns_self(self, tiny_train):
        ranker = PopularityRanker()
        result = ranker.fit(tiny_train)
        assert result is ranker, "fit() should return self for chaining"

    def test_popularity_counts_correct(self, fitted_pop):
        pop = fitted_pop._pop
        assert pop[1] == 3
        assert pop[2] == 2
        assert pop[3] == 1

    def test_score_before_fit_raises(self):
        ranker = PopularityRanker()
        with pytest.raises(RuntimeError, match="fit"):
            ranker.score(np.array([0]), np.array([[1, 2, 3]]))


class TestPopularityRankerScore:
    def test_output_shape_matches_input(self, fitted_pop):
        user_ids = np.array([0, 1, 2])
        item_ids = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        scores = fitted_pop.score(user_ids, item_ids)
        assert scores.shape == (3, 3)

    def test_scores_reflect_popularity(self, fitted_pop):
        """Item 1 (pop=3) must outscore item 2 (pop=2) must outscore item 3 (pop=1)."""
        user_ids = np.array([0])
        item_ids = np.array([[1, 2, 3]])
        scores = fitted_pop.score(user_ids, item_ids)
        assert scores[0, 0] > scores[0, 1] > scores[0, 2]

    def test_unknown_item_gets_zero_score(self, fitted_pop):
        user_ids = np.array([0])
        item_ids = np.array([[999]])   # item never seen in training
        scores = fitted_pop.score(user_ids, item_ids)
        assert scores[0, 0] == 0.0

    def test_output_dtype_is_float(self, fitted_pop):
        user_ids = np.array([0])
        item_ids = np.array([[1, 2]])
        scores = fitted_pop.score(user_ids, item_ids)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_user_agnostic(self, fitted_pop):
        """Scores must be identical for all users (non-personalised model)."""
        item_ids = np.tile([1, 2, 3], (5, 1))   # same items for 5 users
        user_ids = np.arange(5)
        scores = fitted_pop.score(user_ids, item_ids)
        for row in scores[1:]:
            np.testing.assert_array_equal(row, scores[0])

    def test_positive_always_top1_when_most_popular(self, fitted_pop):
        """If the positive (col 0) is the most popular item, HR@1 should be 1.0."""
        from cf_pipeline.eval.metrics import hit_rate_at_k

        # col 0 = item 1 (pop=3), negatives = items 2 and 3 (pop=2,1)
        user_ids = np.array([0])
        item_ids = np.array([[1, 2, 3]])
        scores = fitted_pop.score(user_ids, item_ids)
        hr1 = hit_rate_at_k(scores, k=1)
        assert hr1 == 1.0

    def test_positive_never_top1_when_least_popular(self, fitted_pop):
        """If the positive (col 0) is the least popular item, HR@1 should be 0.0."""
        from cf_pipeline.eval.metrics import hit_rate_at_k

        # col 0 = item 3 (pop=1), negatives = items 1 and 2 (pop=3,2)
        user_ids = np.array([0])
        item_ids = np.array([[3, 1, 2]])
        scores = fitted_pop.score(user_ids, item_ids)
        hr1 = hit_rate_at_k(scores, k=1)
        assert hr1 == 0.0


class TestPopularityRankerEdgeCases:
    def test_single_item_candidate_set(self, fitted_pop):
        user_ids = np.array([0])
        item_ids = np.array([[1]])
        scores = fitted_pop.score(user_ids, item_ids)
        assert scores.shape == (1, 1)
        assert scores[0, 0] == 3.0

    def test_all_unknowns(self, fitted_pop):
        user_ids = np.array([0, 1])
        item_ids = np.array([[100, 200], [300, 400]])
        scores = fitted_pop.score(user_ids, item_ids)
        np.testing.assert_array_equal(scores, np.zeros((2, 2), dtype=np.float32))

    def test_large_batch_shape(self, tiny_train):
        """PopularityRanker must handle a realistic batch (6035 users × 100 cands)."""
        ranker = PopularityRanker().fit(tiny_train)
        n_users, n_cands = 500, 100
        rng = np.random.default_rng(0)
        user_ids = rng.integers(0, 10, size=n_users)
        item_ids = rng.integers(1, 4, size=(n_users, n_cands))
        scores = ranker.score(user_ids, item_ids)
        assert scores.shape == (n_users, n_cands)
