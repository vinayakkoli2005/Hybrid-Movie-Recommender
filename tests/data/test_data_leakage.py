"""Integration tests for data leakage.

These tests require processed parquet files to exist on disk.
Run `python scripts/prepare_data.py` first (Task 10), then:

    pytest tests/data/test_data_leakage.py -v -m integration

Until processed data exists, every test here skips automatically.
"""
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path("data/processed")


# ── helpers ──────────────────────────────────────────────────────────────────

def _require(filename: str):
    path = PROCESSED / filename
    if not path.exists():
        pytest.skip(
            f"{path} not found — run `python scripts/prepare_data.py` first"
        )
    return path


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_no_test_positives_in_train():
    """Every test positive must be absent from the training set."""
    train = pd.read_parquet(_require("train.parquet"))
    test  = pd.read_parquet(_require("test.parquet"))

    train_pairs = set(zip(train["user_id"], train["item_id"]))
    leaks = [
        (u, p)
        for u, p in zip(test["user_id"], test["positive"])
        if (u, p) in train_pairs
    ]
    assert not leaks, (
        f"{len(leaks)} test positives found in train — data leakage detected!\n"
        f"Example: {leaks[:5]}"
    )


@pytest.mark.integration
def test_no_val_positives_in_train():
    """Every val positive must be absent from the training set."""
    train = pd.read_parquet(_require("train.parquet"))
    val   = pd.read_parquet(_require("val.parquet"))

    train_pairs = set(zip(train["user_id"], train["item_id"]))
    leaks = [
        (u, p)
        for u, p in zip(val["user_id"], val["positive"])
        if (u, p) in train_pairs
    ]
    assert not leaks, (
        f"{len(leaks)} val positives found in train — data leakage detected!\n"
        f"Example: {leaks[:5]}"
    )


@pytest.mark.integration
def test_val_positive_not_in_test():
    """Val and test positives must be different items for each user."""
    val  = pd.read_parquet(_require("val.parquet"))
    test = pd.read_parquet(_require("test.parquet"))

    val_pairs  = set(zip(val["user_id"],  val["positive"]))
    test_pairs = set(zip(test["user_id"], test["positive"]))
    overlap = val_pairs & test_pairs
    assert not overlap, (
        f"{len(overlap)} users share the same val and test positive — "
        "split is wrong."
    )


@pytest.mark.integration
def test_test_negatives_disjoint_from_train_history():
    """No test negative may appear in a user's training history."""
    train = pd.read_parquet(_require("train.parquet"))
    test  = pd.read_parquet(_require("test.parquet"))

    history: dict[int, set[int]] = (
        train.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    bad_users = []
    for _, row in test.iterrows():
        leaked = set(row["negatives"]) & history.get(int(row["user_id"]), set())
        if leaked:
            bad_users.append((int(row["user_id"]), leaked))

    assert not bad_users, (
        f"{len(bad_users)} users have negatives overlapping their train history.\n"
        f"Example: {bad_users[:3]}"
    )


@pytest.mark.integration
def test_val_negatives_disjoint_from_train_history():
    """No val negative may appear in a user's training history."""
    train = pd.read_parquet(_require("train.parquet"))
    val   = pd.read_parquet(_require("val.parquet"))

    history: dict[int, set[int]] = (
        train.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    bad_users = []
    for _, row in val.iterrows():
        leaked = set(row["negatives"]) & history.get(int(row["user_id"]), set())
        if leaked:
            bad_users.append((int(row["user_id"]), leaked))

    assert not bad_users, (
        f"{len(bad_users)} users have val negatives overlapping their train history."
    )


@pytest.mark.integration
def test_each_eval_user_has_exactly_99_negatives():
    """Every row in val and test must have exactly 99 negatives."""
    for split_name in ("val.parquet", "test.parquet"):
        df = pd.read_parquet(_require(split_name))
        bad = df[df["negatives"].apply(len) != 99]
        assert len(bad) == 0, (
            f"{split_name}: {len(bad)} users don't have exactly 99 negatives."
        )
