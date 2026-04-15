"""Unit tests for ItemKNNRanker (Task 16).

All tests are self-contained — no disk I/O or processed data required.
They verify:
  1. Mathematical correctness of the similarity computation.
  2. Personalisation invariant (different users must get different scores).
  3. Top-k pruning effect.
  4. Shrinkage effect.
  5. Edge cases: cold users, unseen items, single-item history.
  6. Interface contract: shape, dtype, RuntimeError before fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cf_pipeline.models.baselines import ItemKNNRanker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_train() -> pd.DataFrame:
    """
    5 users, 4 items.
    Items 1 & 2 co-occur for users {0,1,2} → high similarity.
    Items 3 & 4 co-occur for users {3,4}   → high similarity.
    Items 1 and 3 never co-occur            → zero similarity.
    """
    return pd.DataFrame({
        "user_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "item_id": [1, 2, 1, 2, 1, 2, 3, 4, 3, 4],
    })


@pytest.fixture()
def fitted_knn(small_train) -> ItemKNNRanker:
    return ItemKNNRanker(k_neighbors=10, shrinkage=0.0).fit(small_train)


# ---------------------------------------------------------------------------
# 1. Fit correctness
# ---------------------------------------------------------------------------

class TestItemKNNFit:
    def test_fit_returns_self(self, small_train):
        r = ItemKNNRanker()
        assert r.fit(small_train) is r

    def test_internal_index_covers_all_users_and_items(self, fitted_knn, small_train):
        assert len(fitted_knn._user_to_idx) == small_train["user_id"].nunique()
        assert len(fitted_knn._item_to_idx) == small_train["item_id"].nunique()

    def test_similarity_matrix_shape(self, fitted_knn, small_train):
        n_items = small_train["item_id"].nunique()
        assert fitted_knn._item_sim.shape == (n_items, n_items)

    def test_diagonal_is_zero(self, fitted_knn):
        diag = np.diag(fitted_knn._item_sim)
        np.testing.assert_array_equal(diag, 0.0)

    def test_similarity_symmetric(self, fitted_knn):
        sim = fitted_knn._item_sim
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)

    def test_similarity_non_negative(self, fitted_knn):
        assert (fitted_knn._item_sim >= 0).all()

    def test_co_occurring_items_have_high_similarity(self, fitted_knn):
        """Items 1 & 2 share all 3 users → near-maximum cosine similarity."""
        i1 = fitted_knn._item_to_idx[1]
        i2 = fitted_knn._item_to_idx[2]
        sim_12 = fitted_knn._item_sim[i1, i2]
        assert sim_12 > 0.9, f"Expected sim(1,2) ≈ 1.0, got {sim_12:.4f}"

    def test_non_co_occurring_items_have_zero_similarity(self, fitted_knn):
        """Items 1 and 3 never co-occur → cosine = 0."""
        i1 = fitted_knn._item_to_idx[1]
        i3 = fitted_knn._item_to_idx[3]
        sim_13 = fitted_knn._item_sim[i1, i3]
        assert sim_13 == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Shrinkage effect
# ---------------------------------------------------------------------------

class TestShrinkage:
    def test_higher_shrinkage_lowers_similarity(self):
        """Shrinkage ∈ denominator — higher shrinkage → lower similarity value."""
        train = pd.DataFrame({"user_id": [0, 1], "item_id": [1, 1]})
        # Two items both liked by user 0 only → max raw cosine = 1.0
        train2 = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [1, 2, 1, 2],
        })
        low  = ItemKNNRanker(k_neighbors=10, shrinkage=0.0).fit(train2)
        high = ItemKNNRanker(k_neighbors=10, shrinkage=100.0).fit(train2)

        i1 = low._item_to_idx[1]
        i2 = low._item_to_idx[2]

        assert low._item_sim[i1, i2] > high._item_sim[i1, i2]

    def test_zero_shrinkage_equals_cosine(self):
        """With shrinkage=0, result should equal standard cosine similarity."""
        train = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [1, 2, 1, 2],
        })
        r = ItemKNNRanker(k_neighbors=10, shrinkage=0.0).fit(train)
        i1, i2 = r._item_to_idx[1], r._item_to_idx[2]
        # Both items appear for both users → cosine = 1.0
        assert r._item_sim[i1, i2] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Top-k pruning
# ---------------------------------------------------------------------------

class TestTopKPruning:
    def test_pruning_limits_nonzero_neighbours(self):
        """After pruning with k=1, each item has at most 1 non-zero neighbour."""
        train = pd.DataFrame({
            "user_id": [0, 0, 0, 1, 1, 2],
            "item_id": [1, 2, 3, 1, 2, 3],
        })
        r = ItemKNNRanker(k_neighbors=1, shrinkage=0.0).fit(train)
        for row in r._item_sim:
            n_nonzero = (row > 0).sum()
            assert n_nonzero <= 1, f"Expected ≤1 neighbour, got {n_nonzero}"

    def test_pruning_k_ge_n_items_keeps_all(self):
        """k ≥ n_items means no pruning — all similarities remain."""
        train = pd.DataFrame({"user_id": [0, 0], "item_id": [1, 2]})
        r_big_k  = ItemKNNRanker(k_neighbors=1000, shrinkage=0.0).fit(train)
        r_exact  = ItemKNNRanker(k_neighbors=1,    shrinkage=0.0).fit(train)
        # With only 2 items the diagonal is 0 and sim(1,2) is the only entry.
        # Both k=1 and k=1000 should produce the same matrix.
        np.testing.assert_allclose(r_big_k._item_sim, r_exact._item_sim, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Score — personalisation invariant
# ---------------------------------------------------------------------------

class TestItemKNNScore:
    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            ItemKNNRanker().score(np.array([0]), np.array([[1, 2]]))

    def test_output_shape(self, fitted_knn):
        user_ids = np.array([0, 1])
        item_ids = np.array([[1, 2, 3], [2, 3, 4]])
        scores = fitted_knn.score(user_ids, item_ids)
        assert scores.shape == (2, 3)

    def test_output_dtype_float32(self, fitted_knn):
        scores = fitted_knn.score(np.array([0]), np.array([[1, 2]]))
        assert scores.dtype == np.float32

    def test_personalised_different_users_different_scores(self, fitted_knn):
        """
        User 0 likes items {1,2} (cluster A).
        User 3 likes items {3,4} (cluster B).
        For candidate item 2:
          user 0 should score it high (item 2 is similar to his item 1).
          user 3 should score it ~0  (no overlap between clusters A and B).
        """
        user_ids = np.array([0, 3])
        item_ids = np.array([[2, 3], [2, 3]])   # same candidate set, different users
        scores = fitted_knn.score(user_ids, item_ids)

        score_u0_item2 = scores[0, 0]
        score_u3_item2 = scores[1, 0]
        assert score_u0_item2 > score_u3_item2, (
            f"user0 should score item2 higher than user3: "
            f"{score_u0_item2:.4f} vs {score_u3_item2:.4f}"
        )

    def test_cold_user_gets_zero_scores(self, fitted_knn):
        """A user not in training data should receive all-zero scores."""
        user_ids = np.array([999])    # never seen
        item_ids = np.array([[1, 2]])
        scores = fitted_knn.score(user_ids, item_ids)
        np.testing.assert_array_equal(scores, np.zeros((1, 2), dtype=np.float32))

    def test_unseen_item_gets_zero_score(self, fitted_knn):
        """An item not in training data should receive score 0."""
        user_ids = np.array([0])
        item_ids = np.array([[999, 1]])    # item 999 unseen; item 1 seen
        scores = fitted_knn.score(user_ids, item_ids)
        assert scores[0, 0] == 0.0
        assert scores[0, 1] > 0.0

    def test_co_occurring_candidate_scores_higher_than_nonco_occurring(self):
        """
        The plan's canonical test case (from Task 16 spec).
        User 1 likes {10, 20}; user 2 likes {20, 30}.
        For user 1, item 30 co-occurs with item 20 (shared user 2),
        so item 30 should outscore item 99 (completely unknown).
        """
        train = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 20, 30],
        })
        m = ItemKNNRanker(k_neighbors=5, shrinkage=0.0).fit(train)
        user_ids = np.array([1])
        items    = np.array([[30, 99]])
        s = m.score(user_ids, items)
        assert s[0, 0] > s[0, 1], (
            f"item 30 should outscore unknown item 99: {s[0,0]:.4f} vs {s[0,1]:.4f}"
        )

    def test_history_item_self_score(self):
        """
        If the candidate IS an item already in the user's history, its score
        should be high (its neighbours share many users with the user's history).
        An item the user has never seen should score lower.
        """
        train = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [1, 2, 1, 2, 1, 2],
        })
        r = ItemKNNRanker(k_neighbors=10, shrinkage=0.0).fit(train)
        user_ids = np.array([0])
        # col-0 = item 1 (in user 0's history), col-1 = item 3 (unseen)
        # Note: item 3 is not in training data, so it will score 0.
        item_ids = np.array([[2, 3]])  # item 2 → in history, item 3 → unseen
        scores = r.score(user_ids, item_ids)
        assert scores[0, 0] > scores[0, 1]

    def test_all_candidates_unseen_items(self, fitted_knn):
        """All candidates are unseen → entire row should be zeros."""
        user_ids = np.array([0])
        item_ids = np.array([[100, 200, 300]])
        scores = fitted_knn.score(user_ids, item_ids)
        np.testing.assert_array_equal(scores, np.zeros((1, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# 5. Scale test — realistic batch size
# ---------------------------------------------------------------------------

class TestScale:
    def test_realistic_batch_completes(self):
        """Runs without error on a realistic (500 users × 100 candidates) batch."""
        rng = np.random.default_rng(0)
        n_users, n_items = 200, 300
        # Synthetic sparse interactions (5 items per user on average)
        user_col = rng.integers(0, n_users, size=n_users * 5)
        item_col = rng.integers(0, n_items, size=n_users * 5)
        train = pd.DataFrame({"user_id": user_col, "item_id": item_col}).drop_duplicates()

        ranker = ItemKNNRanker(k_neighbors=20, shrinkage=10.0).fit(train)

        test_users = rng.integers(0, n_users, size=500)
        test_items = rng.integers(0, n_items, size=(500, 100))
        scores = ranker.score(test_users, test_items)

        assert scores.shape == (500, 100)
        assert np.isfinite(scores).all()
        assert (scores >= 0.0).all()


# ---------------------------------------------------------------------------
# 6. Integration — eval harness
# ---------------------------------------------------------------------------

class TestItemKNNWithEvalHarness:
    def test_eval_pipeline_runs(self):
        """ItemKNNRanker plugs into eval_pipeline without errors."""
        from cf_pipeline.eval.protocol import eval_pipeline

        train = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [1, 2, 2, 3, 3, 4],
        })
        ranker = ItemKNNRanker(k_neighbors=5, shrinkage=0.0).fit(train)

        eval_set = pd.DataFrame([
            {"user_id": 0, "positive": 2, "negatives": [3, 4]},
            {"user_id": 1, "positive": 3, "negatives": [1, 4]},
        ])
        metrics = eval_pipeline(ranker, eval_set, ks=(1, 2))

        assert "HR@1"   in metrics
        assert "NDCG@2" in metrics
        for v in metrics.values():
            assert 0.0 <= v <= 1.0
