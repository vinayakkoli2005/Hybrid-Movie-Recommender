import pandas as pd
from cf_pipeline.data.binarize import binarize_ratings


def test_keeps_only_positives_above_threshold():
    df = pd.DataFrame({
        "user_id":    [1, 1, 2, 2, 3],
        "item_id":    [10, 20, 10, 30, 40],
        "rating":     [5,  3,  4,  2,  5],
        "timestamp":  [1,  2,  3,  4,  5],
    })
    out = binarize_ratings(df, threshold=4)
    assert len(out) == 3
    assert set(out["item_id"]) == {10, 40}  # user1→10(5★), user2→10(4★), user3→40(5★)
    assert "rating" not in out.columns      # rating column dropped after binarization


def test_threshold_three():
    df = pd.DataFrame({
        "user_id":   [1, 1],
        "item_id":   [1, 2],
        "rating":    [3, 2],
        "timestamp": [1, 2],
    })
    out = binarize_ratings(df, threshold=3)
    assert len(out) == 1
    assert out.iloc[0]["item_id"] == 1


def test_all_below_threshold_returns_empty():
    df = pd.DataFrame({
        "user_id":   [1, 2],
        "item_id":   [1, 2],
        "rating":    [1, 2],
        "timestamp": [1, 2],
    })
    out = binarize_ratings(df, threshold=4)
    assert len(out) == 0
    assert list(out.columns) == ["user_id", "item_id", "timestamp"]


def test_output_columns_exactly():
    df = pd.DataFrame({
        "user_id":   [1],
        "item_id":   [1],
        "rating":    [5],
        "timestamp": [1],
    })
    out = binarize_ratings(df)
    assert list(out.columns) == ["user_id", "item_id", "timestamp"]


def test_index_is_reset():
    df = pd.DataFrame({
        "user_id":   [1, 1, 1],
        "item_id":   [1, 2, 3],
        "rating":    [5, 1, 4],
        "timestamp": [1, 2, 3],
    })
    out = binarize_ratings(df)
    assert list(out.index) == list(range(len(out)))
