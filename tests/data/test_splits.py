import pandas as pd
import pytest
from cf_pipeline.data.splits import leave_one_out_split


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_df(user_ids, item_ids, timestamps):
    return pd.DataFrame(
        {"user_id": user_ids, "item_id": item_ids, "timestamp": timestamps}
    )


# ── tests ────────────────────────────────────────────────────────────────────

def test_split_holds_out_latest_per_user():
    df = _make_df(
        user_ids=  [1, 1, 1, 1, 2, 2, 2],
        item_ids=  [10, 20, 30, 40, 50, 60, 70],
        timestamps=[1,  2,  3,  4, 10, 20, 30],
    )
    train, val, test = leave_one_out_split(df)
    # user 1: latest=40 (test), 2nd=30 (val), rest={10,20} train
    assert set(train[train["user_id"] == 1]["item_id"]) == {10, 20}
    assert val[val["user_id"] == 1]["item_id"].iloc[0] == 30
    assert test[test["user_id"] == 1]["item_id"].iloc[0] == 40
    # user 2: latest=70 (test), 2nd=60 (val), rest={50} train
    assert set(train[train["user_id"] == 2]["item_id"]) == {50}
    assert val[val["user_id"] == 2]["item_id"].iloc[0] == 60
    assert test[test["user_id"] == 2]["item_id"].iloc[0] == 70


def test_drops_users_with_too_few_interactions():
    df = _make_df([1, 1], [10, 20], [1, 2])
    train, val, test = leave_one_out_split(df, min_interactions=3)
    assert len(train) == 0
    assert len(val) == 0
    assert len(test) == 0


def test_exact_min_interactions_border():
    # user with exactly min_interactions=3 should be included
    df = _make_df([1, 1, 1], [10, 20, 30], [1, 2, 3])
    train, val, test = leave_one_out_split(df, min_interactions=3)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
    assert train.iloc[0]["item_id"] == 10
    assert val.iloc[0]["item_id"] == 20
    assert test.iloc[0]["item_id"] == 30


def test_train_val_test_are_disjoint():
    df = _make_df(
        [1, 1, 1, 1, 1],
        [10, 20, 30, 40, 50],
        [1, 2, 3, 4, 5],
    )
    train, val, test = leave_one_out_split(df)
    train_items = set(zip(train["user_id"], train["item_id"]))
    val_items   = set(zip(val["user_id"],   val["item_id"]))
    test_items  = set(zip(test["user_id"],  test["item_id"]))
    assert train_items.isdisjoint(val_items),  "train ∩ val not empty"
    assert train_items.isdisjoint(test_items), "train ∩ test not empty"
    assert val_items.isdisjoint(test_items),   "val ∩ test not empty"


def test_all_indices_are_reset():
    df = _make_df([1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6], [1, 2, 3, 1, 2, 3])
    train, val, test = leave_one_out_split(df)
    assert list(train.index) == list(range(len(train)))
    assert list(val.index)   == list(range(len(val)))
    assert list(test.index)  == list(range(len(test)))


def test_timestamp_ties_broken_deterministically():
    # Two items have same timestamp — split must be stable across repeated calls
    df = _make_df([1, 1, 1], [10, 20, 30], [1, 2, 2])  # items 20&30 tie at t=2
    train1, val1, test1 = leave_one_out_split(df)
    train2, val2, test2 = leave_one_out_split(df)
    assert test1["item_id"].iloc[0] == test2["item_id"].iloc[0]
    assert val1["item_id"].iloc[0]  == val2["item_id"].iloc[0]


def test_multi_user_coverage():
    # Every eligible user appears exactly once in val and test
    rows = []
    for u in range(1, 11):
        for i, t in enumerate(range(1, 6)):   # 5 interactions each
            rows.append({"user_id": u, "item_id": u * 10 + i, "timestamp": t})
    df = pd.DataFrame(rows)
    train, val, test = leave_one_out_split(df)
    assert val["user_id"].nunique()  == 10
    assert test["user_id"].nunique() == 10
    assert (val.groupby("user_id").size() == 1).all()
    assert (test.groupby("user_id").size() == 1).all()
