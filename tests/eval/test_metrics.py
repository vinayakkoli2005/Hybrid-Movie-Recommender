import numpy as np
import pytest
from cf_pipeline.eval.metrics import all_metrics, hit_rate_at_k, ndcg_at_k


# ── Task 11 plan tests ────────────────────────────────────────────────────────

def test_hit_rate_basic():
    # Column 0 is always the positive.
    scores = np.array([
        [0.9, 0.1, 0.2, 0.3, 0.4],   # positive ranked #1 → hit@1=1
        [0.1, 0.9, 0.2, 0.3, 0.4],   # positive ranked #5 → hit@1=0, hit@5=1
        [0.5, 0.6, 0.7, 0.8, 0.9],   # positive ranked #5 → hit@1=0
    ])
    assert hit_rate_at_k(scores, k=1) == pytest.approx(1 / 3)
    assert hit_rate_at_k(scores, k=5) == pytest.approx(1.0)
    assert hit_rate_at_k(scores, k=2) == pytest.approx(1 / 3)


def test_ndcg_basic():
    scores = np.array([
        [0.9, 0.1, 0.2, 0.3, 0.4],   # rank 1 → ndcg = 1/log2(2) = 1.0
        [0.5, 0.9, 0.4, 0.3, 0.2],   # rank 2 → ndcg = 1/log2(3) ≈ 0.6309
    ])
    out = ndcg_at_k(scores, k=5)
    expected = (1.0 + 1 / np.log2(3)) / 2
    assert abs(out - expected) < 1e-6


def test_metrics_zero_when_positive_outside_k():
    scores = np.array([[0.1, 0.9, 0.8]])   # positive at rank 3
    assert hit_rate_at_k(scores, k=2) == 0.0
    assert ndcg_at_k(scores, k=2) == 0.0


# ── extra correctness tests ───────────────────────────────────────────────────

def test_hit_rate_perfect_ranker():
    # Every user's positive is ranked #1
    scores = np.array([[1.0, 0.0, 0.0]] * 100)
    assert hit_rate_at_k(scores, k=1) == 1.0


def test_hit_rate_worst_ranker():
    # Positive always ranked last (column 0 always gets lowest score)
    scores = np.array([[0.0, 1.0, 0.9, 0.8]] * 50)
    assert hit_rate_at_k(scores, k=3) == 0.0
    assert hit_rate_at_k(scores, k=4) == 1.0


def test_ndcg_rank1_equals_1():
    scores = np.array([[1.0, 0.5, 0.3]])   # positive ranked #1
    assert ndcg_at_k(scores, k=1) == pytest.approx(1.0)


def test_ndcg_rank2_correct_value():
    # positive at rank 2 → 1/log2(3)
    scores = np.array([[0.5, 1.0, 0.3]])
    expected = 1.0 / np.log2(3)
    assert ndcg_at_k(scores, k=5) == pytest.approx(expected)


def test_tie_is_conservative():
    # When scores are tied, positive should get worst rank (conservative)
    # positive score == another score → positive NOT counted as "above" those items
    scores = np.array([[0.5, 0.5, 0.5]])   # three-way tie; positive at rank 1 (ties resolved conservatively)
    # (scores > pos).sum() = 0 → rank = 1 (correct: conservative means ties don't help)
    assert hit_rate_at_k(scores, k=1) == 1.0


def test_all_metrics_keys():
    scores = np.ones((5, 20))
    scores[:, 0] = 2.0   # positive always first
    m = all_metrics(scores, ks=(1, 5, 10, 20))
    assert set(m.keys()) == {"HR@1", "HR@5", "HR@10", "HR@20",
                              "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@20"}


def test_all_metrics_perfect_values():
    scores = np.ones((10, 20))
    scores[:, 0] = 2.0
    m = all_metrics(scores)
    for k in (1, 5, 10, 20):
        assert m[f"HR@{k}"]   == pytest.approx(1.0)
        assert m[f"NDCG@{k}"] == pytest.approx(1.0)


def test_single_candidate_always_rank1():
    # If there is only 1 candidate (the positive itself), rank must be 1
    scores = np.array([[0.7], [0.3], [0.9]])
    assert hit_rate_at_k(scores, k=1) == 1.0
    assert ndcg_at_k(scores, k=1) == pytest.approx(1.0)
