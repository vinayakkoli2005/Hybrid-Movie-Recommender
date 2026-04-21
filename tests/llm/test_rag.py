import pandas as pd
from cf_pipeline.llm.rag import (
    DenseItemIndex,
    BM25ItemIndex,
    reciprocal_rank_fusion,
    build_hyde_query_prompt,
)

ITEMS = pd.DataFrame({
    "item_id": [1, 2, 3],
    "title": ["Toy Story", "Cars", "Saw"],
    "overview": ["Animated toys come alive", "Race cars travel and learn", "Horror puzzle trap"],
    "genres": ["Animation", "Animation", "Horror"],
})


# ── Dense index ──────────────────────────────────────────────────────────────

def test_dense_index_topk():
    idx = DenseItemIndex(model_name="sentence-transformers/all-MiniLM-L6-v2").build(ITEMS)
    top = idx.search("animated movie about toys", k=2)
    assert top[0][0] == 1   # Toy Story
    assert len(top) == 2


def test_dense_index_search_by_id():
    idx = DenseItemIndex(model_name="sentence-transformers/all-MiniLM-L6-v2").build(ITEMS)
    # Toy Story's nearest neighbour should be Cars (both Animation), not itself
    top = idx.search_by_id(1, k=1)
    assert top[0][0] != 1
    assert len(top) == 1


# ── BM25 index ───────────────────────────────────────────────────────────────

def test_bm25_index():
    idx = BM25ItemIndex().build(ITEMS)
    top = idx.search("toys animation", k=2)
    assert top[0][0] == 1   # Toy Story


def test_bm25_returns_k_results():
    idx = BM25ItemIndex().build(ITEMS)
    top = idx.search("horror", k=2)
    assert len(top) == 2


# ── RRF ──────────────────────────────────────────────────────────────────────

def test_rrf_combines_two_lists():
    a = [(1, 0.9), (2, 0.5), (3, 0.1)]
    b = [(2, 0.95), (1, 0.6), (4, 0.2)]
    fused = reciprocal_rank_fusion([a, b], k=3)
    ids = [x[0] for x in fused]
    assert set(ids[:2]) == {1, 2}


def test_rrf_single_list_preserves_order():
    lst = [(10, 1.0), (20, 0.5), (30, 0.1)]
    fused = reciprocal_rank_fusion([lst], k=3)
    assert [x[0] for x in fused] == [10, 20, 30]


# ── HyDE prompt ──────────────────────────────────────────────────────────────

def test_hyde_prompt_includes_history_and_candidate():
    history = [{"title": "Toy Story", "genres": "Animation"}]
    candidate = {"title": "Inside Out", "genres": "Animation", "overview": "Emotions in a girl's mind"}
    p = build_hyde_query_prompt(history, candidate)
    assert "Toy Story" in p
    assert "Inside Out" in p


def test_hyde_prompt_no_history():
    p = build_hyde_query_prompt([], {"title": "Inception"})
    assert "(no history)" in p
    assert "Inception" in p
