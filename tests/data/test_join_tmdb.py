import pandas as pd
import pytest
from cf_pipeline.data.join_tmdb import join_movies_with_tmdb


# ── fixtures ────────────────────────────────────────────────────────────────

def _make_movies():
    return pd.DataFrame({
        "item_id": [1, 2, 3],
        "title":   ["A", "B", "C"],
        "genres":  ["x", "y", "z"],
    })


def _make_links():
    return pd.DataFrame({
        "item_id": [1, 2, 3],
        "imdb_id": ["tt0001", "tt0002", "tt9999"],
    })


def _make_tmdb():
    return pd.DataFrame({
        "imdb_id":  ["tt0001", "tt0002"],
        "overview": ["plot1", "plot2"],
    })


# ── tests ────────────────────────────────────────────────────────────────────

def test_join_keeps_only_matched_items():
    out = join_movies_with_tmdb(_make_movies(), _make_links(), _make_tmdb())
    assert len(out) == 2          # tt9999 not in tmdb → dropped
    assert "overview" in out.columns
    assert set(out["item_id"]) == {1, 2}


def test_join_drops_item_with_no_link():
    # item 3 has no entry in links at all
    movies = pd.DataFrame({"item_id": [1, 2, 3], "title": ["A","B","C"], "genres":["x","y","z"]})
    links  = pd.DataFrame({"item_id": [1, 2],    "imdb_id": ["tt0001","tt0002"]})
    tmdb   = pd.DataFrame({"imdb_id": ["tt0001","tt0002"], "overview":["p1","p2"]})
    out = join_movies_with_tmdb(movies, links, tmdb)
    assert len(out) == 2
    assert 3 not in out["item_id"].values


def test_join_index_is_reset():
    out = join_movies_with_tmdb(_make_movies(), _make_links(), _make_tmdb())
    assert list(out.index) == list(range(len(out)))


def test_join_preserves_all_tmdb_columns():
    tmdb = pd.DataFrame({
        "imdb_id":     ["tt0001"],
        "overview":    ["plot1"],
        "popularity":  [7.5],
        "vote_average":[8.0],
    })
    links  = pd.DataFrame({"item_id": [1], "imdb_id": ["tt0001"]})
    movies = pd.DataFrame({"item_id": [1], "title": ["A"], "genres": ["x"]})
    out = join_movies_with_tmdb(movies, links, tmdb)
    assert "overview"     in out.columns
    assert "popularity"   in out.columns
    assert "vote_average" in out.columns


def test_join_empty_when_no_overlap():
    movies = pd.DataFrame({"item_id": [1], "title": ["A"], "genres": ["x"]})
    links  = pd.DataFrame({"item_id": [1], "imdb_id": ["tt9999"]})
    tmdb   = pd.DataFrame({"imdb_id": ["tt0001"], "overview": ["p"]})
    out = join_movies_with_tmdb(movies, links, tmdb)
    assert len(out) == 0
