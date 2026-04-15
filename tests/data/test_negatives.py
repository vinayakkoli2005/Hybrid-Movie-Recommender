import pandas as pd
import pytest
from cf_pipeline.data.negatives import sample_negatives


# ── helpers ──────────────────────────────────────────────────────────────────

def _train():
    return pd.DataFrame({
        "user_id":   [1, 1, 2],
        "item_id":   [10, 20, 10],
        "timestamp": [1,  2,  1],
    })


def _eval_set():
    return pd.DataFrame({
        "user_id":   [1, 2],
        "item_id":   [30, 20],
        "timestamp": [5,  5],
    })


ALL_ITEMS = list(range(1, 101))


# ── tests ────────────────────────────────────────────────────────────────────

def test_output_columns():
    out = sample_negatives(_eval_set(), _train(), ALL_ITEMS, n_neg=5, seed=42)
    assert set(out.columns) == {"user_id", "positive", "negatives"}


def test_neg_count_correct():
    out = sample_negatives(_eval_set(), _train(), ALL_ITEMS, n_neg=5, seed=42)
    for _, row in out.iterrows():
        assert len(row["negatives"]) == 5


def test_negatives_not_in_train_history():
    out = sample_negatives(_eval_set(), _train(), ALL_ITEMS, n_neg=5, seed=42)
    for _, row in out.iterrows():
        history = set(_train()[_train()["user_id"] == row["user_id"]]["item_id"])
        assert not (set(row["negatives"]) & history), \
            f"user {row['user_id']}: negatives overlap train history"


def test_positive_not_in_negatives():
    out = sample_negatives(_eval_set(), _train(), ALL_ITEMS, n_neg=5, seed=42)
    for _, row in out.iterrows():
        assert row["positive"] not in row["negatives"], \
            "positive item appears in negatives"


def test_negatives_are_unique_per_user():
    out = sample_negatives(_eval_set(), _train(), ALL_ITEMS, n_neg=10, seed=42)
    for _, row in out.iterrows():
        assert len(set(row["negatives"])) == len(row["negatives"]), \
            "duplicate items within a user's negatives"


def test_seed_reproducibility():
    train    = pd.DataFrame({"user_id": [1], "item_id": [10], "timestamp": [1]})
    eval_set = pd.DataFrame({"user_id": [1], "item_id": [20], "timestamp": [5]})
    items    = list(range(1, 51))
    a = sample_negatives(eval_set, train, items, n_neg=5, seed=42)
    b = sample_negatives(eval_set, train, items, n_neg=5, seed=42)
    assert a.iloc[0]["negatives"] == b.iloc[0]["negatives"]


def test_different_seeds_give_different_results():
    train    = pd.DataFrame({"user_id": [1], "item_id": [10], "timestamp": [1]})
    eval_set = pd.DataFrame({"user_id": [1], "item_id": [20], "timestamp": [5]})
    items    = list(range(1, 200))
    a = sample_negatives(eval_set, train, items, n_neg=50, seed=0)
    b = sample_negatives(eval_set, train, items, n_neg=50, seed=1)
    assert a.iloc[0]["negatives"] != b.iloc[0]["negatives"]


def test_one_row_per_eval_user():
    out = sample_negatives(_eval_set(), _train(), ALL_ITEMS, n_neg=5, seed=42)
    assert len(out) == len(_eval_set())
    assert set(out["user_id"]) == set(_eval_set()["user_id"])


def test_user_with_no_train_history_gets_negatives():
    # user 99 has no train history — all items except positive are candidates
    train    = pd.DataFrame({"user_id": [1], "item_id": [5], "timestamp": [1]})
    eval_set = pd.DataFrame({"user_id": [99], "item_id": [1], "timestamp": [5]})
    items    = list(range(1, 20))
    out = sample_negatives(eval_set, train, items, n_neg=5, seed=42)
    row = out.iloc[0]
    assert len(row["negatives"]) == 5
    assert 1 not in row["negatives"]   # positive excluded


def test_negatives_all_from_item_pool():
    out = sample_negatives(_eval_set(), _train(), ALL_ITEMS, n_neg=5, seed=42)
    item_set = set(ALL_ITEMS)
    for _, row in out.iterrows():
        assert set(row["negatives"]).issubset(item_set), \
            "negatives contain items outside the allowed pool"
