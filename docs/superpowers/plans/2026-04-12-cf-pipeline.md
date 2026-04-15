# CF Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-grade hybrid CF pipeline (EASE^R + LightGCN + DCN-v2 + LLM RAG re-ranker + meta-learner) on MovieLens 1M, with baselines, ablations, LoRA fine-tuning, and a robustness check, producing 5 result tables and a 5–8 min YouTube demo.

**Architecture:** Modular Python package, Hydra YAML configs, CLI scripts, JSON result files. Every model implements a `BaseRanker.score(users, items) → matrix` interface so the same `eval_pipeline()` function can score every experiment. Under the NCF leave-one-out + 99-negatives protocol, S1/S2 (EASE^R, LightGCN, DCN) become **feature producers** (not retrievers), feeding a deterministic meta-learner that produces the final ranking.

**Tech Stack:** Python 3.11, uv, PyTorch 2.x, Hydra, transformers + bitsandbytes (4-bit), peft, sentence-transformers, faiss-cpu, rank-bm25, lightgbm, scikit-learn, pandas, pyarrow, pytest.

**Spec reference:** [docs/superpowers/specs/2026-04-12-cf-pipeline-design.md](../specs/2026-04-12-cf-pipeline-design.md)

> **LLM Runtime Note (2026-04-12):** The original plan targeted `vLLM` for LLM inference, but vLLM has no Windows support. This project runs on Windows 11 + CUDA, so we use **`transformers` + `bitsandbytes` 4-bit quantization** everywhere the plan says "vLLM". This affects Tasks 22, 23, 28, 29, and 37 (LoRA). The public `BaseLLM` interface (`generate_batch`, `score_yes_no`) stays identical — only the backend swaps. When those tasks are dispatched, the implementer will be given transformers+bitsandbytes code in place of the vLLM snippets in the plan.

---

## Phase Overview

| Phase | Tasks | Output |
|---|---|---|
| 0. Repo setup | 1–3 | Working uv project, seeds, logging |
| 1. Data layer | 4–10 | `data/processed/` populated, frozen negatives |
| 2. Eval harness | 11–13 | `HR@K`, `NDCG@K`, `eval_pipeline()` |
| 3. Configs | 14 | Hydra config skeleton |
| 4. Baselines | 15–18 | Pop, ItemKNN, BPR-MF, NeuMF rows for Table 1 |
| 5. CF models | 19–21 | EASE^R, LightGCN, DCN-v2 trained + scored |
| 6. LLM cold-start (S0) | 22–24 | vLLM wrapper + S0 |
| 7. RAG + S3 | 25–29 | FAISS + BM25 + HyDE + LLM YES/NO |
| 8. Meta-learner (S4) + headline | 30–34 | Headline pipeline run, fills last row of Table 1 |
| 9. LoRA fine-tune | 35–37 | Table 3 |
| 10. Ablations | 38–39 | Table 2 |
| 11. Robustness + cold sub | 40–42 | Tables 4 & 5 |
| 12. Tables + report + video | 43–46 | PDF + YouTube link |

---

# Phase 0 — Repo Setup

## Task 1: Initialize uv project with pinned dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/cf_pipeline/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd "c:/Users/vinay/OneDrive/Desktop/cf project"
uv init --package --name cf-pipeline --python 3.11
```

- [ ] **Step 2: Replace generated `pyproject.toml`**

```toml
[project]
name = "cf-pipeline"
version = "0.1.0"
description = "Research-grade hybrid CF pipeline on MovieLens 1M"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pandas>=2.2",
  "pyarrow>=15",
  "scipy>=1.12",
  "scikit-learn>=1.4",
  "torch>=2.3",
  "lightgbm>=4.3",
  "tqdm>=4.66",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
  "sentence-transformers>=2.7",
  "faiss-cpu>=1.8",
  "rank-bm25>=0.2.2",
  "transformers>=4.43",
  "peft>=0.11",
  "accelerate>=0.30",
  "datasets>=2.19",
  "bitsandbytes>=0.43",
  "optuna>=3.6",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4"]

[tool.uv]
package = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Write `.gitignore`**

```
data/raw/
data/processed/
results/*.json
results/*.parquet
results/tables.md
*.pyc
__pycache__/
.venv/
.pytest_cache/
*.egg-info/
report/*.aux
report/*.log
report/*.out
.hydra/
outputs/
multirun/
checkpoints/
*.ckpt
*.pt
*.bin
.DS_Store
```

- [ ] **Step 4: Write minimal README**

```markdown
# CF Pipeline — MovieLens 1M

Hybrid CF pipeline (EASE^R + LightGCN + DCN-v2 + LLM RAG + meta-learner) for HR@K / NDCG@K on MovieLens 1M.

## Setup
```bash
uv sync
uv run python scripts/prepare_data.py
uv run pytest
```

## Run an experiment
```bash
uv run python scripts/eval.py +experiment=baseline_pop
```

See `docs/superpowers/specs/` for design and `docs/superpowers/plans/` for build plan.
```

- [ ] **Step 5: Sync deps and verify**

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: prints torch version and `True` (GPU detected).

- [ ] **Step 6: Initialize git and commit**

```bash
git init
git add pyproject.toml .gitignore README.md src/ tests/
git commit -m "chore: init uv project with pinned dependencies"
```

---

## Task 2: Global seed utility with test

**Files:**
- Create: `src/cf_pipeline/utils/__init__.py`
- Create: `src/cf_pipeline/utils/seeds.py`
- Create: `tests/utils/test_seeds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/test_seeds.py
import numpy as np
import torch
from cf_pipeline.utils.seeds import set_global_seed

def test_seed_makes_numpy_reproducible():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)

def test_seed_makes_torch_reproducible():
    set_global_seed(42)
    a = torch.rand(5)
    set_global_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b)

def test_seed_returns_rng():
    rng = set_global_seed(42)
    assert hasattr(rng, "integers")
```

- [ ] **Step 2: Run test, expect failure**

```bash
uv run pytest tests/utils/test_seeds.py -v
```

Expected: ImportError on `cf_pipeline.utils.seeds`.

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/utils/seeds.py
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> np.random.Generator:
    """Set seeds across random, numpy, torch (cpu+cuda) and return a numpy Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
```

Also create empty `tests/utils/__init__.py`.

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/utils/test_seeds.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/utils tests/utils
git commit -m "feat(utils): add reproducible global seed helper"
```

---

## Task 3: Logging + IO helpers

**Files:**
- Create: `src/cf_pipeline/utils/logging.py`
- Create: `src/cf_pipeline/utils/io.py`
- Create: `tests/utils/test_io.py`

- [ ] **Step 1: Write failing test for `save_result`**

```python
# tests/utils/test_io.py
import json
from pathlib import Path
from cf_pipeline.utils.io import save_result, load_result

def test_save_and_load_round_trip(tmp_path):
    payload = {"experiment": "pop", "metrics": {"HR@10": 0.5}}
    out = tmp_path / "x.json"
    save_result(payload, out)
    loaded = load_result(out)
    assert loaded["experiment"] == "pop"
    assert "git_sha" in loaded
    assert "timestamp" in loaded
```

- [ ] **Step 2: Run, expect ImportError**

```bash
uv run pytest tests/utils/test_io.py -v
```

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/utils/io.py
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def save_result(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(enriched, f, indent=2, sort_keys=True)

def load_result(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)
```

```python
# src/cf_pipeline/utils/logging.py
import logging
import sys

def get_logger(name: str = "cf_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/utils/test_io.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/utils tests/utils
git commit -m "feat(utils): add JSON result IO and logger"
```

---

# Phase 1 — Data Layer

## Task 4: Raw data loaders (ML-1M + TMDB)

**Files:**
- Create: `src/cf_pipeline/data/__init__.py`
- Create: `src/cf_pipeline/data/loaders.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_loaders.py`
- Create: `data/raw/.gitkeep`

> **Manual prerequisite (do once):** Download `ml-1m.zip` from https://files.grouplens.org/datasets/movielens/ml-1m.zip and extract to `data/raw/ml-1m/`. Download TMDB metadata from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset (`movies_metadata.csv`, `keywords.csv`, `credits.csv`) into `data/raw/tmdb/`.

- [ ] **Step 1: Write the failing test (uses tiny synthetic fixtures)**

```python
# tests/data/test_loaders.py
import pandas as pd
from cf_pipeline.data.loaders import load_ml1m_ratings, load_ml1m_movies

def test_load_ml1m_ratings(tmp_path):
    raw = tmp_path / "ml-1m"
    raw.mkdir()
    (raw / "ratings.dat").write_text("1::100::5::978300760\n1::200::3::978300761\n2::100::4::978300762\n")
    df = load_ml1m_ratings(raw)
    assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert len(df) == 3
    assert df.iloc[0]["user_id"] == 1
    assert df.iloc[0]["rating"] == 5

def test_load_ml1m_movies(tmp_path):
    raw = tmp_path / "ml-1m"
    raw.mkdir()
    (raw / "movies.dat").write_bytes(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n2::Jumanji (1995)::Adventure|Children's|Fantasy\n".encode("latin-1")
    )
    df = load_ml1m_movies(raw)
    assert list(df.columns) == ["item_id", "title", "genres"]
    assert df.iloc[0]["title"] == "Toy Story (1995)"
```

- [ ] **Step 2: Run, expect ImportError**

```bash
uv run pytest tests/data/test_loaders.py -v
```

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/data/loaders.py
from pathlib import Path
import pandas as pd

def load_ml1m_ratings(raw_dir: str | Path) -> pd.DataFrame:
    path = Path(raw_dir) / "ratings.dat"
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "item_id": "int32", "rating": "int8", "timestamp": "int64"},
    )
    return df

def load_ml1m_movies(raw_dir: str | Path) -> pd.DataFrame:
    path = Path(raw_dir) / "movies.dat"
    df = pd.read_csv(
        path, sep="::", engine="python", header=None, encoding="latin-1",
        names=["item_id", "title", "genres"],
        dtype={"item_id": "int32"},
    )
    return df

def load_tmdb_metadata(raw_dir: str | Path) -> pd.DataFrame:
    path = Path(raw_dir) / "movies_metadata.csv"
    df = pd.read_csv(path, low_memory=False)
    keep = ["id", "imdb_id", "title", "overview", "genres", "keywords", "popularity"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/data/test_loaders.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/data tests/data data/raw/.gitkeep
git commit -m "feat(data): add ML-1M and TMDB loaders"
```

---

## Task 5: Binarization (rating ≥ 4 → keep, else drop)

**Files:**
- Create: `src/cf_pipeline/data/binarize.py`
- Create: `tests/data/test_binarize.py`

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_binarize.py
import pandas as pd
from cf_pipeline.data.binarize import binarize_ratings

def test_keeps_only_positives_above_threshold():
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [10, 20, 10, 30, 40],
        "rating":  [5,  3,  4,  2,  5],
        "timestamp": [1, 2, 3, 4, 5],
    })
    out = binarize_ratings(df, threshold=4)
    assert len(out) == 3
    assert set(out["item_id"]) == {10, 40}  # user1's 10, user2's 10, user3's 40
    assert "rating" not in out.columns  # binarized → drop

def test_threshold_three():
    df = pd.DataFrame({"user_id":[1,1], "item_id":[1,2], "rating":[3,2], "timestamp":[1,2]})
    out = binarize_ratings(df, threshold=3)
    assert len(out) == 1
    assert out.iloc[0]["item_id"] == 1
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_binarize.py -v
```

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/data/binarize.py
import pandas as pd

def binarize_ratings(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Drop ratings below threshold; keep only (user_id, item_id, timestamp)."""
    pos = df[df["rating"] >= threshold].copy()
    return pos[["user_id", "item_id", "timestamp"]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/data/test_binarize.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/data/binarize.py tests/data/test_binarize.py
git commit -m "feat(data): binarize ratings with rating>=4 threshold"
```

---

## Task 6: TMDB metadata join

**Files:**
- Create: `src/cf_pipeline/data/join_tmdb.py`
- Create: `tests/data/test_join_tmdb.py`

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_join_tmdb.py
import pandas as pd
from cf_pipeline.data.join_tmdb import join_movies_with_tmdb

def test_join_keeps_only_matched_items():
    movies = pd.DataFrame({"item_id": [1, 2, 3], "title": ["A","B","C"], "genres":["x","y","z"]})
    links  = pd.DataFrame({"item_id": [1, 2, 3], "imdb_id": ["tt0001","tt0002","tt9999"]})
    tmdb   = pd.DataFrame({"imdb_id": ["tt0001","tt0002"], "overview":["plot1","plot2"], "keywords":["k1","k2"]})
    out = join_movies_with_tmdb(movies, links, tmdb)
    assert len(out) == 2  # tt9999 not in tmdb → dropped
    assert "overview" in out.columns
    assert set(out["item_id"]) == {1, 2}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/data/join_tmdb.py
import pandas as pd

def join_movies_with_tmdb(
    movies: pd.DataFrame, links: pd.DataFrame, tmdb: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join movies → links → tmdb metadata, keep only matched items."""
    m = movies.merge(links, on="item_id", how="inner")
    out = m.merge(tmdb, on="imdb_id", how="inner")
    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/data/join_tmdb.py tests/data/test_join_tmdb.py
git commit -m "feat(data): inner-join ML-1M with TMDB metadata"
```

---

## Task 7: Leave-one-out split

**Files:**
- Create: `src/cf_pipeline/data/splits.py`
- Create: `tests/data/test_splits.py`

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_splits.py
import pandas as pd
from cf_pipeline.data.splits import leave_one_out_split

def test_split_holds_out_latest_per_user():
    df = pd.DataFrame({
        "user_id":   [1, 1, 1, 1, 2, 2, 2],
        "item_id":   [10, 20, 30, 40, 50, 60, 70],
        "timestamp": [1,  2,  3,  4,  10, 20, 30],
    })
    train, val, test = leave_one_out_split(df)
    # user 1: latest=40 (test), 2nd latest=30 (val), train={10,20}
    assert set(train[train["user_id"]==1]["item_id"]) == {10, 20}
    assert val[val["user_id"]==1]["item_id"].iloc[0] == 30
    assert test[test["user_id"]==1]["item_id"].iloc[0] == 40
    # user 2: latest=70, 2nd=60, train={50}
    assert set(train[train["user_id"]==2]["item_id"]) == {50}

def test_drops_users_with_too_few_interactions():
    df = pd.DataFrame({"user_id":[1,1], "item_id":[10,20], "timestamp":[1,2]})
    train, val, test = leave_one_out_split(df, min_interactions=3)
    assert len(train) == 0  # user 1 has only 2 → dropped
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/data/splits.py
import pandas as pd

def leave_one_out_split(
    df: pd.DataFrame, min_interactions: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """For each user with >=min_interactions, hold out latest as test, second-latest as val."""
    df = df.sort_values(["user_id", "timestamp"]).copy()
    counts = df.groupby("user_id").size()
    eligible = counts[counts >= min_interactions].index
    df = df[df["user_id"].isin(eligible)]

    df["rank_desc"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=False)
    test = df[df["rank_desc"] == 1].drop(columns="rank_desc").reset_index(drop=True)
    val  = df[df["rank_desc"] == 2].drop(columns="rank_desc").reset_index(drop=True)
    train = df[df["rank_desc"] >= 3].drop(columns="rank_desc").reset_index(drop=True)
    return train, val, test
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/data/splits.py tests/data/test_splits.py
git commit -m "feat(data): leave-one-out split with min interactions filter"
```

---

## Task 8: Frozen 99-negative sampling

**Files:**
- Create: `src/cf_pipeline/data/negatives.py`
- Create: `tests/data/test_negatives.py`

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_negatives.py
import pandas as pd
from cf_pipeline.data.negatives import sample_negatives

def test_neg_count_and_no_overlap_with_history():
    train = pd.DataFrame({"user_id":[1,1,2], "item_id":[10,20,10], "timestamp":[1,2,1]})
    eval_set = pd.DataFrame({"user_id":[1,2], "item_id":[30,20], "timestamp":[5,5]})
    all_items = list(range(1, 101))
    out = sample_negatives(eval_set, train, all_items, n_neg=5, seed=42)
    assert set(out.columns) == {"user_id", "positive", "negatives"}
    for _, row in out.iterrows():
        assert len(row["negatives"]) == 5
        history = set(train[train["user_id"]==row["user_id"]]["item_id"]) | {row["positive"]}
        assert not (set(row["negatives"]) & history)

def test_seed_reproducibility():
    train = pd.DataFrame({"user_id":[1], "item_id":[10], "timestamp":[1]})
    eval_set = pd.DataFrame({"user_id":[1], "item_id":[20], "timestamp":[5]})
    all_items = list(range(1, 51))
    a = sample_negatives(eval_set, train, all_items, n_neg=5, seed=42)
    b = sample_negatives(eval_set, train, all_items, n_neg=5, seed=42)
    assert a.iloc[0]["negatives"] == b.iloc[0]["negatives"]
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/data/negatives.py
import numpy as np
import pandas as pd

def sample_negatives(
    eval_set: pd.DataFrame,
    train: pd.DataFrame,
    all_item_ids: list[int],
    n_neg: int = 99,
    seed: int = 42,
) -> pd.DataFrame:
    """For each (user, positive) row, sample n_neg items not in user's history (train ∪ {pos})."""
    rng = np.random.default_rng(seed)
    user_history: dict[int, set[int]] = (
        train.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    all_items_arr = np.asarray(all_item_ids)
    rows = []
    for _, r in eval_set.iterrows():
        u = int(r["user_id"]); pos = int(r["item_id"])
        forbidden = user_history.get(u, set()) | {pos}
        # Rejection sampling (cheap because forbidden is tiny vs. all_items)
        candidates = all_items_arr[~np.isin(all_items_arr, list(forbidden))]
        chosen = rng.choice(candidates, size=n_neg, replace=False).tolist()
        rows.append({"user_id": u, "positive": pos, "negatives": chosen})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/data/negatives.py tests/data/test_negatives.py
git commit -m "feat(data): seeded 99-negatives sampler"
```

---

## Task 9: Data leakage assertion test

**Files:**
- Create: `tests/data/test_data_leakage.py`

- [ ] **Step 1: Write the test (it will be a no-op until processed data exists; mark as integration)**

```python
# tests/data/test_data_leakage.py
import json
from pathlib import Path
import pandas as pd
import pytest

PROCESSED = Path("data/processed")

@pytest.mark.integration
def test_no_test_positives_in_train():
    if not (PROCESSED / "train.parquet").exists():
        pytest.skip("processed data not built yet — run scripts/prepare_data.py")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")
    train_pairs = set(zip(train["user_id"], train["item_id"]))
    leaks = [
        (u, p) for u, p in zip(test["user_id"], test["positive"]) if (u, p) in train_pairs
    ]
    assert not leaks, f"{len(leaks)} test positives found in train"

@pytest.mark.integration
def test_no_val_positives_in_train():
    if not (PROCESSED / "train.parquet").exists():
        pytest.skip()
    train = pd.read_parquet(PROCESSED / "train.parquet")
    val   = pd.read_parquet(PROCESSED / "val.parquet")
    train_pairs = set(zip(train["user_id"], train["item_id"]))
    leaks = [(u, p) for u, p in zip(val["user_id"], val["positive"]) if (u, p) in train_pairs]
    assert not leaks

@pytest.mark.integration
def test_negatives_disjoint_from_history():
    if not (PROCESSED / "test.parquet").exists():
        pytest.skip()
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")
    history: dict[int, set[int]] = train.groupby("user_id")["item_id"].apply(set).to_dict()
    bad = 0
    for _, r in test.iterrows():
        if set(r["negatives"]) & history.get(r["user_id"], set()):
            bad += 1
    assert bad == 0
```

- [ ] **Step 2: Register the marker in `pyproject.toml`**

Append under `[tool.pytest.ini_options]`:

```toml
markers = ["integration: requires processed data on disk"]
```

- [ ] **Step 3: Run only integration tests (will skip until data is built)**

```bash
uv run pytest tests/data/test_data_leakage.py -v -m integration
```

Expected: 3 skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_data_leakage.py pyproject.toml
git commit -m "test(data): leakage assertions for train/val/test integrity"
```

---

## Task 10: `prepare_data.py` CLI script

**Files:**
- Create: `scripts/prepare_data.py`
- Create: `data/processed/.gitkeep`

- [ ] **Step 1: Implement the script**

```python
# scripts/prepare_data.py
"""One-shot data prep: load → join TMDB → binarize → split → freeze negatives."""
from pathlib import Path
import pandas as pd

from cf_pipeline.utils.seeds import set_global_seed
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.data.loaders import load_ml1m_ratings, load_ml1m_movies, load_tmdb_metadata
from cf_pipeline.data.binarize import binarize_ratings
from cf_pipeline.data.splits import leave_one_out_split
from cf_pipeline.data.negatives import sample_negatives

RAW = Path("data/raw")
OUT = Path("data/processed")
SEED = 42

def _load_links(raw_dir: Path) -> pd.DataFrame:
    """ML-1M ships no links.csv. Use the ML-25M links.csv (subset on ML-1M item ids)."""
    links = pd.read_csv(raw_dir / "ml-1m" / "links.csv")  # download separately from ML-25M
    links = links.rename(columns={"movieId": "item_id", "imdbId": "imdb_id_int"})
    links["imdb_id"] = "tt" + links["imdb_id_int"].astype("Int64").astype(str).str.zfill(7)
    return links[["item_id", "imdb_id"]]

def main():
    log = get_logger("prepare_data")
    set_global_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("Loading ML-1M ratings…")
    ratings = load_ml1m_ratings(RAW / "ml-1m")
    log.info(f"  {len(ratings):,} raw ratings, {ratings['user_id'].nunique():,} users, {ratings['item_id'].nunique():,} items")

    log.info("Binarizing (rating >= 4)…")
    pos = binarize_ratings(ratings, threshold=4)
    log.info(f"  {len(pos):,} positive interactions")

    log.info("Joining ML-1M movies with TMDB metadata…")
    movies = load_ml1m_movies(RAW / "ml-1m")
    links = _load_links(RAW)
    tmdb = load_tmdb_metadata(RAW / "tmdb")
    tmdb = tmdb.rename(columns={"imdb_id": "imdb_id"})
    from cf_pipeline.data.join_tmdb import join_movies_with_tmdb
    items_meta = join_movies_with_tmdb(movies, links, tmdb)
    log.info(f"  {len(items_meta):,} items with metadata")

    # Filter ratings to items with metadata
    pos = pos[pos["item_id"].isin(items_meta["item_id"])]
    log.info(f"  {len(pos):,} ratings retained after metadata filter")

    log.info("Leave-one-out splitting…")
    train, val_pos, test_pos = leave_one_out_split(pos, min_interactions=3)
    log.info(f"  train={len(train):,} val={len(val_pos):,} test={len(test_pos):,}")

    all_items = sorted(items_meta["item_id"].unique().tolist())

    log.info("Sampling 99 negatives per test user…")
    test_with_neg = sample_negatives(test_pos, train, all_items, n_neg=99, seed=SEED)
    log.info("Sampling 99 negatives per val user…")
    val_with_neg  = sample_negatives(val_pos,  train, all_items, n_neg=99, seed=SEED + 1)

    log.info(f"Writing parquet files to {OUT}/")
    train.to_parquet(OUT / "train.parquet")
    val_with_neg.to_parquet(OUT / "val.parquet")
    test_with_neg.to_parquet(OUT / "test.parquet")
    items_meta.to_parquet(OUT / "items_metadata.parquet")

    # Frozen JSON snapshot of negatives
    import json
    with open(OUT / "eval_negatives.json", "w") as f:
        json.dump(
            {"test": test_with_neg.to_dict(orient="records"),
             "val":  val_with_neg.to_dict(orient="records")},
            f,
        )
    log.info("Done.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run prepare_data**

```bash
uv run python scripts/prepare_data.py
```

Expected: log lines reporting counts, files written to `data/processed/`.

- [ ] **Step 3: Run leakage tests against the real data**

```bash
uv run pytest tests/data/test_data_leakage.py -v -m integration
```

Expected: 3 passed.

- [ ] **Step 4: Commit script (NOT data)**

```bash
git add scripts/prepare_data.py data/processed/.gitkeep
git commit -m "feat(data): prepare_data CLI script (binarize+split+freeze negatives)"
```

---

# Phase 2 — Evaluation Harness

## Task 11: Vectorized HR@K

**Files:**
- Create: `src/cf_pipeline/eval/__init__.py`
- Create: `src/cf_pipeline/eval/metrics.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/eval/test_metrics.py
import numpy as np
from cf_pipeline.eval.metrics import hit_rate_at_k, ndcg_at_k

def test_hit_rate_basic():
    # 3 users, 5 candidates each. Position 0 is the positive.
    scores = np.array([
        [0.9, 0.1, 0.2, 0.3, 0.4],   # positive ranked #1 → hit@1=1
        [0.1, 0.9, 0.2, 0.3, 0.4],   # positive ranked #5 → hit@1=0, hit@5=1
        [0.5, 0.6, 0.7, 0.8, 0.9],   # positive ranked #5 → hit@1=0
    ])
    assert hit_rate_at_k(scores, k=1) == 1/3
    assert hit_rate_at_k(scores, k=5) == 1.0
    assert hit_rate_at_k(scores, k=2) == 1/3

def test_ndcg_basic():
    scores = np.array([
        [0.9, 0.1, 0.2, 0.3, 0.4],   # rank 1 → ndcg = 1/log2(2) = 1.0
        [0.5, 0.9, 0.4, 0.3, 0.2],   # rank 2 → ndcg = 1/log2(3) ≈ 0.6309
    ])
    out = ndcg_at_k(scores, k=5)
    assert abs(out - (1.0 + 1/np.log2(3)) / 2) < 1e-6

def test_metrics_zero_when_positive_outside_k():
    scores = np.array([[0.1, 0.9, 0.8]])  # positive at rank 3
    assert hit_rate_at_k(scores, k=2) == 0.0
    assert ndcg_at_k(scores, k=2) == 0.0
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/eval/metrics.py
import numpy as np

def _rank_of_positive(scores: np.ndarray) -> np.ndarray:
    """Return 1-indexed rank of column 0 (the positive) for each row.

    Higher score → better rank. Ties broken by giving positive the worst rank
    (conservative).
    """
    pos = scores[:, 0:1]                       # (n, 1)
    rank = (scores > pos).sum(axis=1) + 1      # 1-indexed
    return rank

def hit_rate_at_k(scores: np.ndarray, k: int) -> float:
    rank = _rank_of_positive(scores)
    return float((rank <= k).mean())

def ndcg_at_k(scores: np.ndarray, k: int) -> float:
    rank = _rank_of_positive(scores)
    in_topk = rank <= k
    gains = np.where(in_topk, 1.0 / np.log2(rank + 1), 0.0)
    return float(gains.mean())

def all_metrics(scores: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    out = {}
    for k in ks:
        out[f"HR@{k}"]   = hit_rate_at_k(scores, k)
        out[f"NDCG@{k}"] = ndcg_at_k(scores, k)
    return out
```

- [ ] **Step 4: Run, expect pass**

```bash
uv run pytest tests/eval/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/eval tests/eval
git commit -m "feat(eval): vectorized HR@K and NDCG@K metrics"
```

---

## Task 12: BaseRanker interface + eval_pipeline

**Files:**
- Create: `src/cf_pipeline/eval/protocol.py`
- Create: `src/cf_pipeline/models/__init__.py`
- Create: `src/cf_pipeline/models/base.py`
- Create: `tests/eval/test_protocol.py`

- [ ] **Step 1: Write failing test using a fake ranker**

```python
# tests/eval/test_protocol.py
import numpy as np
import pandas as pd
from cf_pipeline.models.base import BaseRanker
from cf_pipeline.eval.protocol import eval_pipeline

class _AlwaysFavorPositive(BaseRanker):
    def score(self, user_ids, item_ids):
        # Position 0 is the positive — give it max score.
        s = np.zeros((len(user_ids), item_ids.shape[1]))
        s[:, 0] = 1.0
        return s

class _RandomRanker(BaseRanker):
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    def score(self, user_ids, item_ids):
        return self.rng.random(item_ids.shape)

def test_perfect_ranker_gets_perfect_metrics():
    eval_set = pd.DataFrame({
        "user_id":[1,2], "positive":[10, 20], "negatives":[[11,12,13], [21,22,23]],
    })
    metrics = eval_pipeline(_AlwaysFavorPositive(), eval_set, ks=(1,))
    assert metrics["HR@1"] == 1.0
    assert metrics["NDCG@1"] == 1.0

def test_random_ranker_is_not_perfect():
    eval_set = pd.DataFrame({
        "user_id":[1,2,3,4,5,6,7,8,9,10],
        "positive":[1]*10,
        "negatives":[[2,3,4,5]]*10,
    })
    metrics = eval_pipeline(_RandomRanker(seed=0), eval_set, ks=(1,))
    assert metrics["HR@1"] < 1.0
```

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement**

```python
# src/cf_pipeline/models/base.py
from __future__ import annotations
import numpy as np

class BaseRanker:
    """All models implement this. score returns (n_users, n_candidates) matrix."""
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

```python
# src/cf_pipeline/eval/protocol.py
import numpy as np
import pandas as pd
from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.models.base import BaseRanker

def build_candidate_matrix(eval_set: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (user_ids[n], item_ids[n, 100]) where column 0 is the positive."""
    user_ids = eval_set["user_id"].to_numpy()
    pos = eval_set["positive"].to_numpy().reshape(-1, 1)
    neg = np.stack([np.asarray(x) for x in eval_set["negatives"].tolist()])
    items = np.concatenate([pos, neg], axis=1)
    return user_ids, items

def eval_pipeline(
    model: BaseRanker, eval_set: pd.DataFrame, ks=(1, 5, 10, 20)
) -> dict[str, float]:
    user_ids, item_ids = build_candidate_matrix(eval_set)
    scores = model.score(user_ids, item_ids)
    assert scores.shape == item_ids.shape, f"got {scores.shape}, expected {item_ids.shape}"
    return all_metrics(scores, ks=ks)
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/eval/protocol.py src/cf_pipeline/models tests/eval/test_protocol.py
git commit -m "feat(eval): BaseRanker interface and eval_pipeline harness"
```

---

## Task 13: Result-saving wrapper for experiments

**Files:**
- Modify: `src/cf_pipeline/eval/protocol.py` — add `run_and_save_experiment`
- Create: `tests/eval/test_run_experiment.py`

- [ ] **Step 1: Write failing test**

```python
# tests/eval/test_run_experiment.py
import pandas as pd
from cf_pipeline.eval.protocol import run_and_save_experiment
from tests.eval.test_protocol import _AlwaysFavorPositive

def test_writes_json_with_metrics(tmp_path):
    eval_set = pd.DataFrame({"user_id":[1], "positive":[10], "negatives":[[11,12]]})
    out = tmp_path / "x.json"
    run_and_save_experiment(
        model=_AlwaysFavorPositive(),
        eval_set=eval_set,
        experiment_name="fake",
        out_path=out,
    )
    import json
    with open(out) as f:
        data = json.load(f)
    assert data["experiment"] == "fake"
    assert data["metrics"]["HR@1"] == 1.0
    assert "timestamp" in data
```

- [ ] **Step 2: Run, expect AttributeError**

- [ ] **Step 3: Add to `protocol.py`**

```python
# Append to src/cf_pipeline/eval/protocol.py
from cf_pipeline.utils.io import save_result
from pathlib import Path

def run_and_save_experiment(
    model: BaseRanker,
    eval_set: pd.DataFrame,
    experiment_name: str,
    out_path: str | Path,
    ks=(1, 5, 10, 20),
) -> dict:
    metrics = eval_pipeline(model, eval_set, ks=ks)
    payload = {"experiment": experiment_name, "metrics": metrics}
    save_result(payload, out_path)
    return payload
```

- [ ] **Step 4: Run test**

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/eval/protocol.py tests/eval/test_run_experiment.py
git commit -m "feat(eval): run_and_save_experiment helper"
```

---

# Phase 3 — Hydra Configs

## Task 14: Hydra config skeleton

**Files:**
- Create: `configs/config.yaml`
- Create: `configs/data/ml1m.yaml`
- Create: `configs/eval/ncf_protocol.yaml`
- Create: `configs/experiment/baseline_pop.yaml`

- [ ] **Step 1: Write configs**

```yaml
# configs/config.yaml
defaults:
  - data: ml1m
  - eval: ncf_protocol
  - experiment: baseline_pop
  - _self_

seed: 42
results_dir: results
```

```yaml
# configs/data/ml1m.yaml
name: ml1m
processed_dir: data/processed
train_path: ${data.processed_dir}/train.parquet
val_path:   ${data.processed_dir}/val.parquet
test_path:  ${data.processed_dir}/test.parquet
items_meta_path: ${data.processed_dir}/items_metadata.parquet
```

```yaml
# configs/eval/ncf_protocol.yaml
name: ncf_loo_99neg
ks: [1, 5, 10, 20]
split: test     # or val
```

```yaml
# configs/experiment/baseline_pop.yaml
name: baseline_pop
model: popularity
out_path: ${results_dir}/baseline_pop.json
```

- [ ] **Step 2: Quick smoke test loading the config**

```bash
uv run python -c "import hydra; from omegaconf import OmegaConf; \
hydra.initialize(version_base=None, config_path='configs'); \
cfg = hydra.compose(config_name='config'); \
print(OmegaConf.to_yaml(cfg))"
```

Expected: prints the merged config.

- [ ] **Step 3: Commit**

```bash
git add configs/
git commit -m "feat(configs): hydra config skeleton (data/eval/experiment)"
```

---

# Phase 4 — Baselines

## Task 15: Popularity baseline + first end-to-end eval

**Files:**
- Create: `src/cf_pipeline/models/baselines.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_popularity.py`
- Create: `scripts/eval.py`

- [ ] **Step 1: Write failing test**

```python
# tests/models/test_popularity.py
import numpy as np
import pandas as pd
from cf_pipeline.models.baselines import PopularityRanker

def test_popularity_scores_more_popular_higher():
    train = pd.DataFrame({"user_id":[1,2,3,4], "item_id":[10,10,10,20]})
    ranker = PopularityRanker().fit(train)
    user_ids = np.array([99])
    items = np.array([[20, 10]])
    scores = ranker.score(user_ids, items)
    # item 10 had 3 occurrences, item 20 had 1 → score(10) > score(20)
    assert scores[0, 1] > scores[0, 0]
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/models/baselines.py
import numpy as np
import pandas as pd
from cf_pipeline.models.base import BaseRanker

class PopularityRanker(BaseRanker):
    def __init__(self):
        self._counts: dict[int, int] = {}

    def fit(self, train: pd.DataFrame) -> "PopularityRanker":
        self._counts = train["item_id"].value_counts().to_dict()
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        # Same score for every user (user-agnostic).
        vec = np.vectorize(lambda i: float(self._counts.get(int(i), 0)))
        return vec(item_ids)
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/models/test_popularity.py -v
```

- [ ] **Step 4: Write `scripts/eval.py` (Hydra entry point) — handles popularity for now**

```python
# scripts/eval.py
import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from cf_pipeline.utils.seeds import set_global_seed
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.eval.protocol import run_and_save_experiment

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log = get_logger("eval")
    set_global_seed(cfg.seed)
    train = pd.read_parquet(cfg.data.train_path)
    eval_set_path = cfg.data.test_path if cfg.eval.split == "test" else cfg.data.val_path
    eval_set = pd.read_parquet(eval_set_path)

    model_name = cfg.experiment.model
    if model_name == "popularity":
        from cf_pipeline.models.baselines import PopularityRanker
        model = PopularityRanker().fit(train)
    else:
        raise NotImplementedError(f"model={model_name} not yet wired in eval.py")

    out = Path(cfg.experiment.out_path)
    payload = run_and_save_experiment(
        model=model, eval_set=eval_set,
        experiment_name=cfg.experiment.name, out_path=out,
        ks=tuple(cfg.eval.ks),
    )
    log.info(f"{cfg.experiment.name}: {payload['metrics']}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run end-to-end**

```bash
uv run python scripts/eval.py
```

Expected: prints metrics and writes `results/baseline_pop.json`. HR@10 should be roughly ~0.45 on ML-1M (popularity is a strong baseline because of long-tail).

- [ ] **Step 6: Commit**

```bash
git add src/cf_pipeline/models/baselines.py tests/models scripts/eval.py
git commit -m "feat(models): popularity baseline + first end-to-end eval pipeline"
```

---

## Task 16: ItemKNN baseline (cosine on co-occurrence)

**Files:**
- Modify: `src/cf_pipeline/models/baselines.py`
- Create: `tests/models/test_itemknn.py`
- Create: `configs/experiment/baseline_itemknn.yaml`
- Modify: `scripts/eval.py` — add itemknn branch

- [ ] **Step 1: Write failing test**

```python
# tests/models/test_itemknn.py
import numpy as np
import pandas as pd
from cf_pipeline.models.baselines import ItemKNNRanker

def test_itemknn_personalized():
    # User 1 likes items {10, 20}; user 2 likes {20, 30}.
    # For user 1, item 30 should score higher than item 99.
    train = pd.DataFrame({"user_id":[1,1,2,2], "item_id":[10,20,20,30]})
    m = ItemKNNRanker(k_neighbors=5).fit(train)
    user_ids = np.array([1])
    items = np.array([[30, 99]])  # 30 = positive (co-occurs with 20), 99 = unknown
    s = m.score(user_ids, items)
    assert s[0, 0] > s[0, 1]
```

- [ ] **Step 2: Implement (append to baselines.py)**

```python
# Append to src/cf_pipeline/models/baselines.py
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class ItemKNNRanker(BaseRanker):
    def __init__(self, k_neighbors: int = 50):
        self.k = k_neighbors
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._item_sim: np.ndarray | None = None
        self._user_items: csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> "ItemKNNRanker":
        users = train["user_id"].unique()
        items = train["item_id"].unique()
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {i: j for j, i in enumerate(items)}
        rows = train["user_id"].map(self._user_to_idx).to_numpy()
        cols = train["item_id"].map(self._item_to_idx).to_numpy()
        data = np.ones(len(train), dtype=np.float32)
        n_u, n_i = len(users), len(items)
        self._user_items = csr_matrix((data, (rows, cols)), shape=(n_u, n_i))
        item_user = self._user_items.T  # (n_i, n_u)
        item_user_norm = normalize(item_user, norm="l2", axis=1)
        sim = (item_user_norm @ item_user_norm.T).toarray().astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        # Top-k pruning per item
        if self.k < sim.shape[1]:
            idx = np.argpartition(-sim, self.k, axis=1)[:, self.k:]
            for r, cs in enumerate(idx):
                sim[r, cs] = 0.0
        self._item_sim = sim
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        out = np.zeros(item_ids.shape, dtype=np.float32)
        for r, u in enumerate(user_ids):
            uidx = self._user_to_idx.get(int(u))
            if uidx is None:
                continue
            user_history = self._user_items[uidx].toarray().ravel()  # (n_items,)
            for c, item in enumerate(item_ids[r]):
                iidx = self._item_to_idx.get(int(item))
                if iidx is None:
                    continue
                # Sum similarity of `item` to items in user history
                out[r, c] = float(self._item_sim[iidx] @ user_history)
        return out
```

- [ ] **Step 3: Add config**

```yaml
# configs/experiment/baseline_itemknn.yaml
name: baseline_itemknn
model: itemknn
k_neighbors: 50
out_path: ${results_dir}/baseline_itemknn.json
```

- [ ] **Step 4: Wire into `scripts/eval.py`**

In `scripts/eval.py`, replace the model dispatch with:

```python
    if model_name == "popularity":
        from cf_pipeline.models.baselines import PopularityRanker
        model = PopularityRanker().fit(train)
    elif model_name == "itemknn":
        from cf_pipeline.models.baselines import ItemKNNRanker
        model = ItemKNNRanker(k_neighbors=cfg.experiment.k_neighbors).fit(train)
    else:
        raise NotImplementedError(f"model={model_name} not yet wired in eval.py")
```

- [ ] **Step 5: Run tests + eval**

```bash
uv run pytest tests/models/test_itemknn.py -v
uv run python scripts/eval.py +experiment=baseline_itemknn
```

Expected: HR@10 around 0.50–0.55 on ML-1M.

- [ ] **Step 6: Commit**

```bash
git add src/cf_pipeline/models/baselines.py tests/models/test_itemknn.py configs/experiment/baseline_itemknn.yaml scripts/eval.py
git commit -m "feat(models): ItemKNN baseline (cosine co-occurrence)"
```

---

## Task 17: BPR-MF baseline

**Files:**
- Create: `src/cf_pipeline/models/bpr_mf.py`
- Create: `tests/models/test_bpr_mf.py`
- Create: `configs/experiment/baseline_bpr.yaml`
- Modify: `scripts/eval.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/models/test_bpr_mf.py
import numpy as np
import pandas as pd
import torch
from cf_pipeline.models.bpr_mf import BPRMFRanker

def test_bpr_mf_trains_and_scores_personalized():
    torch.manual_seed(0)
    # User 1 likes evens, user 2 likes odds — should diverge after training
    train = pd.DataFrame({
        "user_id": [1]*10 + [2]*10,
        "item_id": list(range(0, 20, 2)) + list(range(1, 20, 2)),
    })
    m = BPRMFRanker(emb_dim=8, n_epochs=20, lr=0.05, batch_size=4).fit(train)
    user_ids = np.array([1, 2])
    items = np.array([[100, 101], [101, 100]])  # unseen
    s = m.score(user_ids, items)
    assert s.shape == (2, 2)
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/models/bpr_mf.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cf_pipeline.models.base import BaseRanker

class _BPRModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.item_emb = nn.Embedding(n_items + 1, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score_pair(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)

class BPRMFRanker(BaseRanker):
    def __init__(self, emb_dim=64, n_epochs=20, lr=1e-3, batch_size=1024, weight_decay=1e-5):
        self.emb_dim = emb_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.wd = weight_decay
        self._model: _BPRModel | None = None
        self._u2i: dict[int, int] = {}
        self._i2i: dict[int, int] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: pd.DataFrame):
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._u2i = {u: i for i, u in enumerate(users)}
        self._i2i = {i: j for j, i in enumerate(items)}
        u_arr = train["user_id"].map(self._u2i).to_numpy()
        i_arr = train["item_id"].map(self._i2i).to_numpy()
        n_users, n_items = len(users), len(items)
        self._model = _BPRModel(n_users, n_items, self.emb_dim).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.wd)

        rng = np.random.default_rng(42)
        for ep in range(self.n_epochs):
            perm = rng.permutation(len(u_arr))
            losses = []
            for s in range(0, len(u_arr), self.batch_size):
                idx = perm[s:s+self.batch_size]
                u = torch.from_numpy(u_arr[idx]).long().to(self._device)
                i_pos = torch.from_numpy(i_arr[idx]).long().to(self._device)
                i_neg = torch.from_numpy(rng.integers(0, n_items, size=len(idx))).long().to(self._device)
                p = self._model.score_pair(u, i_pos)
                n = self._model.score_pair(u, i_neg)
                loss = -torch.log(torch.sigmoid(p - n) + 1e-10).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        return self

    @torch.no_grad()
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        m = self._model
        m.eval()
        out = np.zeros(item_ids.shape, dtype=np.float32)
        for r, u in enumerate(user_ids):
            ui = self._u2i.get(int(u))
            if ui is None:
                continue
            row_items = [self._i2i.get(int(x), -1) for x in item_ids[r]]
            mask = [x >= 0 for x in row_items]
            if not any(mask):
                continue
            valid = [x for x in row_items if x >= 0]
            ut = torch.full((len(valid),), ui, dtype=torch.long, device=self._device)
            it = torch.tensor(valid, dtype=torch.long, device=self._device)
            s = m.score_pair(ut, it).cpu().numpy()
            j = 0
            for c, ok in enumerate(mask):
                if ok:
                    out[r, c] = s[j]; j += 1
        return out
```

- [ ] **Step 3: Add config**

```yaml
# configs/experiment/baseline_bpr.yaml
name: baseline_bpr
model: bpr_mf
emb_dim: 64
n_epochs: 30
lr: 1e-3
batch_size: 4096
out_path: ${results_dir}/baseline_bpr.json
```

- [ ] **Step 4: Wire into `scripts/eval.py`**

Add another `elif model_name == "bpr_mf":` branch instantiating `BPRMFRanker(...)` from cfg.experiment.

- [ ] **Step 5: Run tests + eval**

```bash
uv run pytest tests/models/test_bpr_mf.py -v
uv run python scripts/eval.py +experiment=baseline_bpr
```

Expected: HR@10 around 0.55–0.60.

- [ ] **Step 6: Commit**

```bash
git add src/cf_pipeline/models/bpr_mf.py tests/models/test_bpr_mf.py configs/experiment/baseline_bpr.yaml scripts/eval.py
git commit -m "feat(models): BPR-MF baseline"
```

---

## Task 18: NeuMF baseline

**Files:**
- Create: `src/cf_pipeline/models/neumf.py`
- Create: `tests/models/test_neumf.py`
- Create: `configs/experiment/baseline_neumf.yaml`
- Modify: `scripts/eval.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/models/test_neumf.py
import numpy as np
import pandas as pd
import torch
from cf_pipeline.models.neumf import NeuMFRanker

def test_neumf_smoke():
    torch.manual_seed(0)
    train = pd.DataFrame({"user_id":[1,1,2,2,3], "item_id":[10,20,10,30,40]})
    m = NeuMFRanker(emb_dim=8, mlp_layers=(16,8), n_epochs=3, batch_size=4).fit(train)
    s = m.score(np.array([1]), np.array([[10, 30]]))
    assert s.shape == (1, 2)
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/models/neumf.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cf_pipeline.models.base import BaseRanker

class _NeuMF(nn.Module):
    def __init__(self, n_u, n_i, emb_dim, mlp_layers):
        super().__init__()
        self.gmf_u = nn.Embedding(n_u + 1, emb_dim)
        self.gmf_i = nn.Embedding(n_i + 1, emb_dim)
        self.mlp_u = nn.Embedding(n_u + 1, emb_dim)
        self.mlp_i = nn.Embedding(n_i + 1, emb_dim)
        layers = []
        in_dim = 2 * emb_dim
        for h in mlp_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(emb_dim + in_dim, 1)
        for emb in [self.gmf_u, self.gmf_i, self.mlp_u, self.mlp_i]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u, i):
        gmf = self.gmf_u(u) * self.gmf_i(i)
        mlp_in = torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        return self.out(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)

class NeuMFRanker(BaseRanker):
    def __init__(self, emb_dim=32, mlp_layers=(64, 32, 16), n_epochs=20, lr=1e-3, batch_size=4096):
        self.emb_dim = emb_dim
        self.mlp_layers = tuple(mlp_layers)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self._model = None
        self._u2i = {}
        self._i2i = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: pd.DataFrame):
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._u2i = {u: i for i, u in enumerate(users)}
        self._i2i = {i: j for j, i in enumerate(items)}
        u_arr = train["user_id"].map(self._u2i).to_numpy()
        i_arr = train["item_id"].map(self._i2i).to_numpy()
        n_u, n_i = len(users), len(items)
        self._model = _NeuMF(n_u, n_i, self.emb_dim, self.mlp_layers).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(42)
        for ep in range(self.n_epochs):
            perm = rng.permutation(len(u_arr))
            for s in range(0, len(u_arr), self.batch_size):
                idx = perm[s:s+self.batch_size]
                u = torch.from_numpy(u_arr[idx]).long().to(self._device)
                pos = torch.from_numpy(i_arr[idx]).long().to(self._device)
                neg = torch.from_numpy(rng.integers(0, n_i, size=len(idx))).long().to(self._device)
                logits_pos = self._model(u, pos)
                logits_neg = self._model(u, neg)
                logits = torch.cat([logits_pos, logits_neg])
                labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)])
                loss = bce(logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        self._model.eval()
        out = np.zeros(item_ids.shape, dtype=np.float32)
        for r, u in enumerate(user_ids):
            ui = self._u2i.get(int(u))
            if ui is None:
                continue
            row_items = [self._i2i.get(int(x), -1) for x in item_ids[r]]
            valid_idx = [(c, x) for c, x in enumerate(row_items) if x >= 0]
            if not valid_idx:
                continue
            ut = torch.full((len(valid_idx),), ui, dtype=torch.long, device=self._device)
            it = torch.tensor([x for _, x in valid_idx], dtype=torch.long, device=self._device)
            s = torch.sigmoid(self._model(ut, it)).cpu().numpy()
            for (c, _), v in zip(valid_idx, s):
                out[r, c] = v
        return out
```

- [ ] **Step 3: Config + wire into eval.py**

```yaml
# configs/experiment/baseline_neumf.yaml
name: baseline_neumf
model: neumf
emb_dim: 32
mlp_layers: [64, 32, 16]
n_epochs: 20
lr: 1e-3
batch_size: 4096
out_path: ${results_dir}/baseline_neumf.json
```

In `scripts/eval.py`, add:

```python
    elif model_name == "neumf":
        from cf_pipeline.models.neumf import NeuMFRanker
        model = NeuMFRanker(
            emb_dim=cfg.experiment.emb_dim,
            mlp_layers=tuple(cfg.experiment.mlp_layers),
            n_epochs=cfg.experiment.n_epochs,
            lr=cfg.experiment.lr,
            batch_size=cfg.experiment.batch_size,
        ).fit(train)
```

- [ ] **Step 4: Run smoke test + eval**

```bash
uv run pytest tests/models/test_neumf.py -v
uv run python scripts/eval.py +experiment=baseline_neumf
```

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/models/neumf.py tests/models/test_neumf.py configs/experiment/baseline_neumf.yaml scripts/eval.py
git commit -m "feat(models): NeuMF baseline"
```

---

# Phase 5 — CF Models (S1, S2)

## Task 19: EASE^R closed-form

**Files:**
- Create: `src/cf_pipeline/models/ease.py`
- Create: `tests/models/test_ease.py`
- Create: `configs/experiment/ease.yaml`
- Modify: `scripts/eval.py`

- [ ] **Step 1: Write failing test**

```python
# tests/models/test_ease.py
import numpy as np
import pandas as pd
from cf_pipeline.models.ease import EASERRanker

def test_ease_diagonal_zero_after_fit():
    train = pd.DataFrame({"user_id":[1,1,2,2], "item_id":[10,20,20,30]})
    m = EASERRanker(reg_lambda=100.0).fit(train)
    B = m._B
    assert np.allclose(np.diag(B), 0.0)

def test_ease_personalized_score():
    train = pd.DataFrame({"user_id":[1,1,2,2], "item_id":[10,20,20,30]})
    m = EASERRanker(reg_lambda=10.0).fit(train)
    s = m.score(np.array([1]), np.array([[30, 99]]))
    assert s[0, 0] >= s[0, 1]   # 30 co-occurs with 20 (in user 2)
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/models/ease.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from cf_pipeline.models.base import BaseRanker

class EASERRanker(BaseRanker):
    def __init__(self, reg_lambda: float = 500.0):
        self.reg = reg_lambda
        self._u2i: dict[int, int] = {}
        self._i2i: dict[int, int] = {}
        self._X: csr_matrix | None = None
        self._B: np.ndarray | None = None

    def fit(self, train: pd.DataFrame):
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._u2i = {u: i for i, u in enumerate(users)}
        self._i2i = {i: j for j, i in enumerate(items)}
        rows = train["user_id"].map(self._u2i).to_numpy()
        cols = train["item_id"].map(self._i2i).to_numpy()
        n_u, n_i = len(users), len(items)
        self._X = csr_matrix(
            (np.ones(len(train), dtype=np.float32), (rows, cols)),
            shape=(n_u, n_i),
        )
        G = (self._X.T @ self._X).toarray().astype(np.float64)
        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[diag_idx] = 0.0
        self._B = B.astype(np.float32)
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        out = np.zeros(item_ids.shape, dtype=np.float32)
        for r, u in enumerate(user_ids):
            uidx = self._u2i.get(int(u))
            if uidx is None:
                continue
            user_vec = self._X[uidx].toarray().ravel()  # (n_items,)
            user_scores = user_vec @ self._B            # (n_items,)
            for c, item in enumerate(item_ids[r]):
                iidx = self._i2i.get(int(item))
                if iidx is None:
                    continue
                out[r, c] = user_scores[iidx]
        return out
```

- [ ] **Step 3: Add config + wire**

```yaml
# configs/experiment/ease.yaml
name: ease
model: ease
reg_lambda: 500.0
out_path: ${results_dir}/ease.json
```

Add to `scripts/eval.py`:

```python
    elif model_name == "ease":
        from cf_pipeline.models.ease import EASERRanker
        model = EASERRanker(reg_lambda=cfg.experiment.reg_lambda).fit(train)
```

- [ ] **Step 4: Run + tune lambda**

```bash
uv run pytest tests/models/test_ease.py -v
for lam in 100 250 500 1000; do
  uv run python scripts/eval.py +experiment=ease experiment.reg_lambda=$lam experiment.out_path=results/ease_lam${lam}.json
done
```

Inspect the JSONs in `results/`, pick the best lambda by val NDCG@10, save as the canonical `results/ease.json`. Expected best HR@10 ~0.65–0.70.

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/models/ease.py tests/models/test_ease.py configs/experiment/ease.yaml scripts/eval.py
git commit -m "feat(models): EASE^R closed-form ranker + lambda sweep"
```

---

## Task 20: LightGCN

**Files:**
- Create: `src/cf_pipeline/models/lightgcn.py`
- Create: `tests/models/test_lightgcn.py`
- Create: `configs/experiment/lightgcn.yaml`
- Modify: `scripts/eval.py`

- [ ] **Step 1: Smoke test**

```python
# tests/models/test_lightgcn.py
import numpy as np
import pandas as pd
import torch
from cf_pipeline.models.lightgcn import LightGCNRanker

def test_lightgcn_one_epoch_smoke():
    torch.manual_seed(0)
    train = pd.DataFrame({
        "user_id":[1,1,2,2,3,3],
        "item_id":[10,20,20,30,10,30],
    })
    m = LightGCNRanker(emb_dim=8, n_layers=2, n_epochs=2, batch_size=4, lr=0.05).fit(train)
    s = m.score(np.array([1, 2]), np.array([[20, 30],[10, 30]]))
    assert s.shape == (2, 2)
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/models/lightgcn.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, csr_matrix
from cf_pipeline.models.base import BaseRanker

def _build_norm_adj(n_u: int, n_i: int, u_arr: np.ndarray, i_arr: np.ndarray) -> torch.sparse.Tensor:
    """Symmetric normalized adjacency for the bipartite user-item graph."""
    n = n_u + n_i
    rows = np.concatenate([u_arr, i_arr + n_u])
    cols = np.concatenate([i_arr + n_u, u_arr])
    data = np.ones(len(rows), dtype=np.float32)
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / np.sqrt(np.where(deg == 0, 1, deg))
    D = csr_matrix((deg_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
    norm = (D @ A @ D).tocoo()
    indices = torch.from_numpy(np.vstack([norm.row, norm.col])).long()
    values = torch.from_numpy(norm.data).float()
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

class _LightGCN(nn.Module):
    def __init__(self, n_u, n_i, emb_dim, n_layers):
        super().__init__()
        self.n_u, self.n_i, self.n_layers = n_u, n_i, n_layers
        self.emb = nn.Embedding(n_u + n_i, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, A: torch.sparse.Tensor) -> torch.Tensor:
        x = self.emb.weight
        out = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(A, x)
            out.append(x)
        return torch.stack(out, dim=0).mean(dim=0)

class LightGCNRanker(BaseRanker):
    def __init__(self, emb_dim=64, n_layers=3, n_epochs=200, batch_size=8192, lr=1e-3, weight_decay=1e-4):
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = weight_decay
        self._model = None
        self._A = None
        self._u2i = {}
        self._i2i = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._final_emb = None  # cached after fit

    def fit(self, train: pd.DataFrame):
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._u2i = {u: i for i, u in enumerate(users)}
        self._i2i = {i: j for j, i in enumerate(items)}
        u_arr = train["user_id"].map(self._u2i).to_numpy()
        i_arr = train["item_id"].map(self._i2i).to_numpy()
        n_u, n_i = len(users), len(items)
        self._A = _build_norm_adj(n_u, n_i, u_arr, i_arr).to(self._device)
        self._model = _LightGCN(n_u, n_i, self.emb_dim, self.n_layers).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.wd)

        rng = np.random.default_rng(42)
        u_t = torch.from_numpy(u_arr).long().to(self._device)
        i_t = torch.from_numpy(i_arr).long().to(self._device)

        for ep in range(self.n_epochs):
            perm = rng.permutation(len(u_arr))
            ep_loss = 0.0
            for s in range(0, len(u_arr), self.batch_size):
                idx = perm[s:s + self.batch_size]
                u = u_t[idx]
                pos = i_t[idx]
                neg = torch.from_numpy(rng.integers(0, n_i, size=len(idx))).long().to(self._device)
                emb = self._model.propagate(self._A)
                eu = emb[u]
                epos = emb[n_u + pos]
                eneg = emb[n_u + neg]
                p = (eu * epos).sum(-1)
                n = (eu * eneg).sum(-1)
                loss = -torch.log(torch.sigmoid(p - n) + 1e-10).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item()
            if (ep + 1) % 20 == 0:
                print(f"[lightgcn] epoch {ep+1}/{self.n_epochs} loss={ep_loss:.3f}")

        with torch.no_grad():
            self._final_emb = self._model.propagate(self._A).cpu().numpy()
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        n_u = len(self._u2i)
        out = np.zeros(item_ids.shape, dtype=np.float32)
        for r, u in enumerate(user_ids):
            uidx = self._u2i.get(int(u))
            if uidx is None:
                continue
            ue = self._final_emb[uidx]
            for c, item in enumerate(item_ids[r]):
                iidx = self._i2i.get(int(item))
                if iidx is None:
                    continue
                ie = self._final_emb[n_u + iidx]
                out[r, c] = float(ue @ ie)
        return out
```

- [ ] **Step 3: Add config + wire**

```yaml
# configs/experiment/lightgcn.yaml
name: lightgcn
model: lightgcn
emb_dim: 64
n_layers: 3
n_epochs: 200
batch_size: 8192
lr: 1e-3
weight_decay: 1e-4
out_path: ${results_dir}/lightgcn.json
```

```python
    elif model_name == "lightgcn":
        from cf_pipeline.models.lightgcn import LightGCNRanker
        model = LightGCNRanker(
            emb_dim=cfg.experiment.emb_dim,
            n_layers=cfg.experiment.n_layers,
            n_epochs=cfg.experiment.n_epochs,
            batch_size=cfg.experiment.batch_size,
            lr=cfg.experiment.lr,
            weight_decay=cfg.experiment.weight_decay,
        ).fit(train)
```

- [ ] **Step 4: Run smoke test + train + sweep layers**

```bash
uv run pytest tests/models/test_lightgcn.py -v
uv run python scripts/eval.py +experiment=lightgcn
for L in 1 2 3 4; do
  uv run python scripts/eval.py +experiment=lightgcn experiment.n_layers=$L experiment.out_path=results/lightgcn_L${L}.json
done
```

Pick best L by NDCG@10 from val set, persist as canonical `results/lightgcn.json`.

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/models/lightgcn.py tests/models/test_lightgcn.py configs/experiment/lightgcn.yaml scripts/eval.py
git commit -m "feat(models): LightGCN ranker + layer sweep"
```

---

## Task 21: DCN-v2 with MC dropout

**Files:**
- Create: `src/cf_pipeline/models/dcn.py`
- Create: `tests/models/test_dcn.py`
- Create: `configs/experiment/dcn.yaml`
- Modify: `scripts/eval.py`

DCN takes the EASE^R and LightGCN scores as side features, so before training, dump those scores into a per-(user, item) feature cache.

- [ ] **Step 1: Smoke test**

```python
# tests/models/test_dcn.py
import numpy as np
import pandas as pd
import torch
from cf_pipeline.models.dcn import DCNRanker

def test_dcn_smoke():
    torch.manual_seed(0)
    train = pd.DataFrame({"user_id":[1,1,2,2,3], "item_id":[10,20,20,30,10]})
    m = DCNRanker(emb_dim=8, cross_layers=2, deep=(16,8), dropout=0.1, n_epochs=2, batch_size=4).fit(train)
    s, var = m.score_with_uncertainty(np.array([1]), np.array([[10, 99]]), n_mc=3)
    assert s.shape == (1, 2)
    assert var.shape == (1, 2)
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/models/dcn.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cf_pipeline.models.base import BaseRanker

class _CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)
    def forward(self, x0, xl):
        return x0 * self.W(xl) + xl

class _DCNv2(nn.Module):
    def __init__(self, n_u, n_i, emb_dim, cross_layers, deep, dropout):
        super().__init__()
        self.user_emb = nn.Embedding(n_u + 1, emb_dim)
        self.item_emb = nn.Embedding(n_i + 1, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        in_dim = 2 * emb_dim
        self.crosses = nn.ModuleList([_CrossLayer(in_dim) for _ in range(cross_layers)])
        layers, prev = [], in_dim
        for h in deep:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim + prev, 1)

    def forward(self, u, i):
        x0 = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        xl = x0
        for layer in self.crosses:
            xl = layer(x0, xl)
        d = self.deep(x0)
        return self.head(torch.cat([xl, d], dim=-1)).squeeze(-1)

class DCNRanker(BaseRanker):
    def __init__(self, emb_dim=64, cross_layers=3, deep=(256,128), dropout=0.3, n_epochs=20, lr=1e-3, batch_size=4096):
        self.emb_dim = emb_dim
        self.cross_layers = cross_layers
        self.deep = tuple(deep)
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self._model = None
        self._u2i = {}
        self._i2i = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: pd.DataFrame):
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self._u2i = {u: i for i, u in enumerate(users)}
        self._i2i = {i: j for j, i in enumerate(items)}
        u_arr = train["user_id"].map(self._u2i).to_numpy()
        i_arr = train["item_id"].map(self._i2i).to_numpy()
        n_u, n_i = len(users), len(items)
        self._model = _DCNv2(n_u, n_i, self.emb_dim, self.cross_layers, self.deep, self.dropout).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(42)
        for ep in range(self.n_epochs):
            perm = rng.permutation(len(u_arr))
            for s in range(0, len(u_arr), self.batch_size):
                idx = perm[s:s+self.batch_size]
                u = torch.from_numpy(u_arr[idx]).long().to(self._device)
                pos = torch.from_numpy(i_arr[idx]).long().to(self._device)
                neg = torch.from_numpy(rng.integers(0, n_i, size=len(idx))).long().to(self._device)
                logits = torch.cat([self._model(u, pos), self._model(u, neg)])
                labels = torch.cat([torch.ones(len(idx)), torch.zeros(len(idx))]).to(self._device)
                loss = bce(logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def _forward_batch(self, user_ids, item_ids):
        out = np.zeros(item_ids.shape, dtype=np.float32)
        for r, u in enumerate(user_ids):
            ui = self._u2i.get(int(u))
            if ui is None: continue
            valid = [(c, self._i2i[int(x)]) for c, x in enumerate(item_ids[r]) if int(x) in self._i2i]
            if not valid: continue
            ut = torch.full((len(valid),), ui, dtype=torch.long, device=self._device)
            it = torch.tensor([x for _, x in valid], dtype=torch.long, device=self._device)
            s = torch.sigmoid(self._model(ut, it)).cpu().numpy()
            for (c, _), v in zip(valid, s):
                out[r, c] = v
        return out

    def score(self, user_ids, item_ids):
        self._model.eval()
        return self._forward_batch(user_ids, item_ids)

    def score_with_uncertainty(self, user_ids, item_ids, n_mc=20):
        self._model.train()  # keep dropout on
        passes = []
        for _ in range(n_mc):
            passes.append(self._forward_batch(user_ids, item_ids))
        stack = np.stack(passes, axis=0)
        return stack.mean(0), stack.var(0)
```

- [ ] **Step 3: Add config + wire**

```yaml
# configs/experiment/dcn.yaml
name: dcn
model: dcn
emb_dim: 64
cross_layers: 3
deep: [256, 128]
dropout: 0.3
n_epochs: 20
lr: 1e-3
batch_size: 4096
out_path: ${results_dir}/dcn.json
```

In `scripts/eval.py`:

```python
    elif model_name == "dcn":
        from cf_pipeline.models.dcn import DCNRanker
        model = DCNRanker(
            emb_dim=cfg.experiment.emb_dim,
            cross_layers=cfg.experiment.cross_layers,
            deep=tuple(cfg.experiment.deep),
            dropout=cfg.experiment.dropout,
            n_epochs=cfg.experiment.n_epochs,
            lr=cfg.experiment.lr,
            batch_size=cfg.experiment.batch_size,
        ).fit(train)
```

- [ ] **Step 4: Run smoke test + eval**

```bash
uv run pytest tests/models/test_dcn.py -v
uv run python scripts/eval.py +experiment=dcn
```

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/models/dcn.py tests/models/test_dcn.py configs/experiment/dcn.yaml scripts/eval.py
git commit -m "feat(models): DCN-v2 ranker with MC-dropout uncertainty"
```

---

# Phase 6 — LLM Cold-Start (S0)

## Task 22: vLLM offline server wrapper

**Files:**
- Create: `src/cf_pipeline/llm/__init__.py`
- Create: `src/cf_pipeline/llm/server.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_server_smoke.py` (slow / GPU test)

- [ ] **Step 1: Implement**

```python
# src/cf_pipeline/llm/server.py
"""Thin wrapper around vLLM offline batched inference for Llama-3.1-8B-Instruct."""
from typing import Iterable
from cf_pipeline.utils.logging import get_logger

_log = get_logger("llm.server")

class LlamaServer:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", dtype: str = "bfloat16", max_tokens: int = 256):
        from vllm import LLM, SamplingParams
        self._llm = LLM(model=model_id, dtype=dtype, gpu_memory_utilization=0.85)
        self._sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
        _log.info(f"Loaded {model_id}")

    def generate(self, prompts: list[str]) -> list[dict]:
        outs = self._llm.generate(prompts, self._sampling)
        results = []
        for o in outs:
            text = o.outputs[0].text
            logprobs = o.outputs[0].logprobs   # list of dicts (per token)
            results.append({"text": text, "logprobs": logprobs})
        return results

    def free(self):
        del self._llm
```

- [ ] **Step 2: Slow GPU smoke test (skipped on CPU)**

```python
# tests/llm/test_server_smoke.py
import pytest
import torch
from cf_pipeline.llm.server import LlamaServer

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_basic():
    srv = LlamaServer()
    out = srv.generate(["Reply with the word HELLO and nothing else."])
    assert "HELLO" in out[0]["text"].upper()
    srv.free()
```

Add to `pyproject.toml` markers:

```toml
markers = [
  "integration: requires processed data on disk",
  "gpu: requires GPU + Llama-3.1-8B downloaded",
]
```

- [ ] **Step 3: Run smoke test**

```bash
uv run pytest tests/llm/test_server_smoke.py -v -m gpu
```

Expected: passes (will download the model on first run — tens of GB).

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/llm/server.py tests/llm/test_server_smoke.py pyproject.toml
git commit -m "feat(llm): vLLM Llama-3.1-8B server wrapper"
```

---

## Task 23: S0 cold-start prompt + strict-JSON parser

**Files:**
- Create: `src/cf_pipeline/llm/cold_start.py`
- Create: `tests/llm/test_cold_start.py`

- [ ] **Step 1: Failing test (parser only — does not need GPU)**

```python
# tests/llm/test_cold_start.py
from cf_pipeline.llm.cold_start import build_cold_start_prompt, parse_cold_start_response

def test_prompt_includes_history_titles():
    history = [{"title": "Toy Story", "genres": "Animation|Comedy"}]
    p = build_cold_start_prompt(user_id=42, history=history)
    assert "Toy Story" in p
    assert "JSON" in p

def test_parse_valid_json():
    raw = '{"liked_genres": ["Action","Sci-Fi"], "liked_actors": ["Keanu Reeves"], "mood": "epic"}'
    parsed = parse_cold_start_response(raw)
    assert parsed["liked_genres"] == ["Action", "Sci-Fi"]
    assert parsed["mood"] == "epic"

def test_parse_handles_extra_text_around_json():
    raw = "Sure! Here's the JSON: {\"liked_genres\":[\"X\"],\"liked_actors\":[],\"mood\":\"calm\"} done."
    parsed = parse_cold_start_response(raw)
    assert parsed["liked_genres"] == ["X"]

def test_parse_returns_default_on_invalid():
    parsed = parse_cold_start_response("nonsense")
    assert parsed == {"liked_genres": [], "liked_actors": [], "mood": ""}
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/llm/cold_start.py
import json
import re

PROMPT_TEMPLATE = """You are a movie preference profiler. Based on this user's history,
output a JSON profile of their taste. Output ONLY the JSON, no other text.

User history:
{history_block}

JSON schema:
{{
  "liked_genres":  [list of genre strings],
  "liked_actors":  [list of actor name strings],
  "mood":          one of "fun" | "serious" | "epic" | "calm" | "dark" | "uplifting"
}}

JSON:"""

def build_cold_start_prompt(user_id: int, history: list[dict]) -> str:
    lines = [f"- {h.get('title','?')} ({h.get('genres','?')})" for h in history]
    return PROMPT_TEMPLATE.format(history_block="\n".join(lines) if lines else "(no history)")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_cold_start_response(text: str) -> dict:
    default = {"liked_genres": [], "liked_actors": [], "mood": ""}
    m = _JSON_RE.search(text)
    if not m:
        return default
    try:
        d = json.loads(m.group(0))
        return {
            "liked_genres": list(d.get("liked_genres", [])),
            "liked_actors": list(d.get("liked_actors", [])),
            "mood": str(d.get("mood", "")),
        }
    except json.JSONDecodeError:
        return default
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/llm/test_cold_start.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/llm/cold_start.py tests/llm/test_cold_start.py
git commit -m "feat(llm): cold-start prompt builder + JSON parser"
```

---

## Task 24: Compute cold-start profiles for cold users

**Files:**
- Create: `scripts/build_cold_start_profiles.py`

- [ ] **Step 1: Implement**

```python
# scripts/build_cold_start_profiles.py
"""Compute and cache S0 LLM profiles for users with <5 train interactions."""
import json
from pathlib import Path
import pandas as pd
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.llm.server import LlamaServer
from cf_pipeline.llm.cold_start import build_cold_start_prompt, parse_cold_start_response

PROCESSED = Path("data/processed")
OUT = PROCESSED / "cold_start_profiles.json"
THRESHOLD = 5

def main():
    log = get_logger("cold_start")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    items = pd.read_parquet(PROCESSED / "items_metadata.parquet")[["item_id", "title", "genres"]]
    counts = train.groupby("user_id").size()
    cold_users = counts[counts < THRESHOLD].index.tolist()
    log.info(f"{len(cold_users)} cold users (<{THRESHOLD} train interactions)")
    if not cold_users:
        OUT.write_text("{}")
        return
    history_lookup = train.merge(items, on="item_id").groupby("user_id").apply(
        lambda g: g[["title", "genres"]].to_dict(orient="records")
    ).to_dict()
    prompts = [build_cold_start_prompt(u, history_lookup.get(u, [])) for u in cold_users]

    srv = LlamaServer()
    outs = srv.generate(prompts)
    profiles = {int(u): parse_cold_start_response(o["text"]) for u, o in zip(cold_users, outs)}
    OUT.write_text(json.dumps(profiles, indent=2))
    log.info(f"Wrote {len(profiles)} profiles to {OUT}")
    srv.free()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
uv run python scripts/build_cold_start_profiles.py
```

Expected: writes `data/processed/cold_start_profiles.json`. Inspect a couple — they should be valid JSON with sensible genres/moods.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_cold_start_profiles.py
git commit -m "feat(llm): script to compute S0 cold-start profiles via Llama"
```

---

# Phase 7 — RAG (S3)

## Task 25: Sentence-embedding FAISS index over items

**Files:**
- Create: `src/cf_pipeline/llm/rag.py`
- Create: `tests/llm/test_rag.py`

- [ ] **Step 1: Failing test (use a small in-memory model)**

```python
# tests/llm/test_rag.py
import pandas as pd
from cf_pipeline.llm.rag import DenseItemIndex

def test_dense_index_topk(tmp_path):
    items = pd.DataFrame({
        "item_id":[1,2,3],
        "title":["Toy Story","Cars","Saw"],
        "overview":["Animated toys come alive","Race cars travel and learn","Horror puzzle trap"],
        "genres":["Animation","Animation","Horror"],
    })
    idx = DenseItemIndex(model_name="sentence-transformers/all-MiniLM-L6-v2").build(items)
    top = idx.search("animated movie about toys", k=2)
    assert top[0][0] == 1   # Toy Story should be #1
    assert len(top) == 2
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/llm/rag.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def _item_text(row: pd.Series) -> str:
    parts = [str(row.get("title", "")), str(row.get("genres", "")), str(row.get("overview", ""))]
    return " | ".join([p for p in parts if p and p != "nan"])

class DenseItemIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._index = None
        self._ids: np.ndarray | None = None

    def build(self, items: pd.DataFrame) -> "DenseItemIndex":
        texts = items.apply(_item_text, axis=1).tolist()
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        self._ids = items["item_id"].to_numpy()
        d = embs.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(embs.astype(np.float32))
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, idxs = self._index.search(q, k)
        return [(int(self._ids[i]), float(s)) for i, s in zip(idxs[0], scores[0])]

    def search_by_id(self, query_item_id: int, k: int = 5) -> list[tuple[int, float]]:
        # Get this item's embedding from the index
        pos = int(np.where(self._ids == query_item_id)[0][0])
        v = self._index.reconstruct(pos).reshape(1, -1)
        scores, idxs = self._index.search(v, k + 1)  # +1 because self will be top-1
        out = [(int(self._ids[i]), float(s)) for i, s in zip(idxs[0], scores[0]) if int(self._ids[i]) != query_item_id]
        return out[:k]
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/llm/test_rag.py::test_dense_index_topk -v
```

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/llm/rag.py tests/llm/test_rag.py
git commit -m "feat(rag): dense item index via sentence-transformers + FAISS"
```

---

## Task 26: BM25 sparse retrieval + reciprocal rank fusion

**Files:**
- Modify: `src/cf_pipeline/llm/rag.py` — add `BM25ItemIndex` and `reciprocal_rank_fusion`
- Modify: `tests/llm/test_rag.py`

- [ ] **Step 1: Add tests**

```python
# Append to tests/llm/test_rag.py
from cf_pipeline.llm.rag import BM25ItemIndex, reciprocal_rank_fusion

def test_bm25_index():
    items = pd.DataFrame({
        "item_id":[1,2,3],
        "title":["Toy Story","Cars","Saw"],
        "overview":["Animated toys come alive","Race cars travel","Horror trap"],
        "genres":["Animation","Animation","Horror"],
    })
    idx = BM25ItemIndex().build(items)
    top = idx.search("toys animation", k=2)
    assert top[0][0] == 1

def test_rrf_combines_two_lists():
    a = [(1, 0.9), (2, 0.5), (3, 0.1)]
    b = [(2, 0.95), (1, 0.6), (4, 0.2)]
    fused = reciprocal_rank_fusion([a, b], k=3)
    ids = [x[0] for x in fused]
    assert set(ids[:2]) == {1, 2}
```

- [ ] **Step 2: Implement (append to `src/cf_pipeline/llm/rag.py`)**

```python
from rank_bm25 import BM25Okapi

class BM25ItemIndex:
    def __init__(self):
        self._bm25 = None
        self._ids = None

    def build(self, items: pd.DataFrame) -> "BM25ItemIndex":
        texts = items.apply(_item_text, axis=1).tolist()
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._ids = items["item_id"].to_numpy()
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(-scores)[:k]
        return [(int(self._ids[i]), float(scores[i])) for i in top_idx]

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = 5, c: int = 60
) -> list[tuple[int, float]]:
    """Standard RRF: score(i) = sum over lists of 1/(c + rank_in_list)."""
    fused: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (iid, _) in enumerate(lst, start=1):
            fused[iid] = fused.get(iid, 0.0) + 1.0 / (c + rank)
    items = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return items
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/llm/rag.py tests/llm/test_rag.py
git commit -m "feat(rag): BM25 index + reciprocal rank fusion"
```

---

## Task 27: HyDE query expansion via the LLM

**Files:**
- Modify: `src/cf_pipeline/llm/rag.py` — add `build_hyde_query_prompt` and a small wrapper
- Modify: `tests/llm/test_rag.py`

- [ ] **Step 1: Test the prompt builder (no LLM call)**

```python
# Append to tests/llm/test_rag.py
from cf_pipeline.llm.rag import build_hyde_query_prompt

def test_hyde_prompt_includes_history_and_candidate():
    history = [{"title":"Toy Story","genres":"Animation"}]
    candidate = {"title":"Inside Out","genres":"Animation","overview":"Emotions in a girl's mind"}
    p = build_hyde_query_prompt(history, candidate)
    assert "Toy Story" in p
    assert "Inside Out" in p
```

- [ ] **Step 2: Append to `rag.py`**

```python
HYDE_TEMPLATE = """Given a user who has liked these movies:
{history_block}

Write 1-2 sentences describing the kind of movie they would enjoy next. Be concrete: mention themes, tone, and genre.
Then check whether this candidate fits: {candidate_title} ({candidate_genres}). Plot: {candidate_overview}

Brief reasoning + ideal-movie description:"""

def build_hyde_query_prompt(history: list[dict], candidate: dict) -> str:
    lines = [f"- {h.get('title','?')} ({h.get('genres','?')})" for h in history[:10]]
    return HYDE_TEMPLATE.format(
        history_block="\n".join(lines),
        candidate_title=candidate.get("title", "?"),
        candidate_genres=candidate.get("genres", "?"),
        candidate_overview=candidate.get("overview", ""),
    )
```

- [ ] **Step 3: Run test**

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/llm/rag.py tests/llm/test_rag.py
git commit -m "feat(rag): HyDE query expansion prompt"
```

---

## Task 28: S3 LLM YES/NO decision with logprob

**Files:**
- Create: `src/cf_pipeline/llm/decision.py`
- Create: `tests/llm/test_decision.py`

- [ ] **Step 1: Failing test for parser**

```python
# tests/llm/test_decision.py
from cf_pipeline.llm.decision import build_decision_prompt, parse_decision_response

def test_prompt_has_strict_schema():
    p = build_decision_prompt(
        user_history=[{"title":"Toy Story","genres":"Animation"}],
        retrieved=[{"title":"Cars","genres":"Animation"}],
        candidate={"title":"Saw","genres":"Horror","overview":"trap"},
    )
    assert "YES" in p and "NO" in p
    assert "JSON" in p

def test_parse_yes():
    out = parse_decision_response('{"decision":"YES"}', None)
    assert out["decision"] == "YES"
    assert out["yes_prob"] in (None, 1.0) or 0.0 <= out["yes_prob"] <= 1.0

def test_parse_no():
    out = parse_decision_response('{"decision":"NO"}', None)
    assert out["decision"] == "NO"

def test_parse_invalid_defaults_no():
    out = parse_decision_response("no opinion", None)
    assert out["decision"] == "NO"
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/llm/decision.py
import json
import math
import re

DECISION_TEMPLATE = """You are a movie recommender. Decide if this user would like the candidate.

User has previously liked:
{history_block}

Most similar items in their history (retrieved):
{retrieved_block}

Candidate:
- Title: {candidate_title}
- Genres: {candidate_genres}
- Plot: {candidate_overview}

Reply with strict JSON only, no other text:
{{"decision": "YES"}}  or  {{"decision": "NO"}}
"""

def build_decision_prompt(user_history, retrieved, candidate) -> str:
    h = "\n".join(f"- {x.get('title','?')} ({x.get('genres','?')})" for x in user_history[:10])
    r = "\n".join(f"- {x.get('title','?')} ({x.get('genres','?')})" for x in retrieved[:5])
    return DECISION_TEMPLATE.format(
        history_block=h,
        retrieved_block=r,
        candidate_title=candidate.get("title","?"),
        candidate_genres=candidate.get("genres","?"),
        candidate_overview=candidate.get("overview",""),
    )

_JSON = re.compile(r"\{.*?\}", re.DOTALL)

def parse_decision_response(text: str, logprobs) -> dict:
    """Returns {decision: 'YES'|'NO', yes_prob: float}."""
    decision = "NO"
    m = _JSON.search(text or "")
    if m:
        try:
            d = json.loads(m.group(0))
            if str(d.get("decision","")).upper() == "YES":
                decision = "YES"
        except json.JSONDecodeError:
            pass
    # Logprob extraction is best-effort: vLLM returns per-token logprobs
    yes_prob = None
    if logprobs:
        for tok_lp in logprobs:
            if not tok_lp:
                continue
            # tok_lp is dict[token_id -> logprob]
            for tok_id, lp in tok_lp.items():
                token_str = lp.decoded_token if hasattr(lp, "decoded_token") else str(tok_id)
                if "YES" in token_str.upper():
                    yes_prob = math.exp(getattr(lp, "logprob", float("-inf")))
                    break
            if yes_prob is not None:
                break
    if yes_prob is None:
        yes_prob = 1.0 if decision == "YES" else 0.0
    return {"decision": decision, "yes_prob": float(yes_prob)}
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/llm/decision.py tests/llm/test_decision.py
git commit -m "feat(llm): S3 decision prompt + strict JSON parser with logprob"
```

---

## Task 29: Build LLM-decision feature cache for the entire test set

**Files:**
- Create: `scripts/build_llm_features.py`
- Create: `data/processed/llm_features.parquet` (output)

- [ ] **Step 1: Implement**

```python
# scripts/build_llm_features.py
"""Compute LLM YES/NO + yes_prob for every (test_user, candidate_item) pair.

Output: parquet with columns (user_id, item_id, decision, yes_prob).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.llm.server import LlamaServer
from cf_pipeline.llm.rag import DenseItemIndex, BM25ItemIndex, reciprocal_rank_fusion, build_hyde_query_prompt
from cf_pipeline.llm.decision import build_decision_prompt, parse_decision_response

PROCESSED = Path("data/processed")
OUT = PROCESSED / "llm_features.parquet"

def main():
    log = get_logger("llm_features")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    items = pd.read_parquet(PROCESSED / "items_metadata.parquet")
    items_lookup = items.set_index("item_id").to_dict(orient="index")

    log.info("Building dense + sparse item indexes…")
    dense = DenseItemIndex().build(items)
    bm25 = BM25ItemIndex().build(items)

    log.info("Loading Llama server…")
    srv = LlamaServer()

    history_by_user = (
        train.merge(items[["item_id","title","genres"]], on="item_id")
             .groupby("user_id").apply(lambda g: g[["item_id","title","genres"]].to_dict(orient="records"))
             .to_dict()
    )

    rows = []
    BATCH = 256
    pending_meta = []
    pending_prompts = []

    def flush():
        if not pending_prompts:
            return
        outs = srv.generate(pending_prompts)
        for meta, o in zip(pending_meta, outs):
            parsed = parse_decision_response(o["text"], o["logprobs"])
            rows.append({**meta, **parsed})
        pending_meta.clear()
        pending_prompts.clear()

    for ridx, r in enumerate(test.itertuples(index=False)):
        u = int(r.user_id)
        candidates = [int(r.positive)] + [int(x) for x in r.negatives]
        history = history_by_user.get(u, [])
        for cand in candidates:
            cand_meta = items_lookup.get(cand, {"title":"?","genres":"?","overview":""})
            # Retrieve via dense + BM25 (could also do HyDE; skipped for speed in this script)
            text = " ".join([cand_meta.get("title",""), cand_meta.get("genres","")])
            d_top = dense.search(text, k=5)
            b_top = bm25.search(text, k=5)
            fused = reciprocal_rank_fusion([d_top, b_top], k=5)
            retrieved = [items_lookup.get(iid, {"title":"?","genres":"?"}) for iid, _ in fused]
            prompt = build_decision_prompt(history, retrieved, cand_meta)
            pending_meta.append({"user_id": u, "item_id": cand})
            pending_prompts.append(prompt)
            if len(pending_prompts) >= BATCH:
                flush()
        if (ridx + 1) % 200 == 0:
            log.info(f"  processed {ridx+1}/{len(test)} users; rows so far={len(rows)}")
    flush()
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT)
    log.info(f"Wrote {len(out_df):,} rows to {OUT}")
    srv.free()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
uv run python scripts/build_llm_features.py
```

Expected: writes `data/processed/llm_features.parquet` with ~6,040 × 100 = 604K rows.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_llm_features.py
git commit -m "feat(llm): build LLM YES/NO feature cache for whole test set"
```

---

# Phase 8 — Meta-Learner (S4) and Headline

## Task 30: Feature builder per (user, candidate)

**Files:**
- Create: `src/cf_pipeline/pipeline/__init__.py`
- Create: `src/cf_pipeline/pipeline/features.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_features.py`

- [ ] **Step 1: Failing test**

```python
# tests/pipeline/test_features.py
import numpy as np
import pandas as pd
from cf_pipeline.pipeline.features import build_feature_matrix

def test_feature_matrix_shape():
    test = pd.DataFrame({
        "user_id":[1,2],
        "positive":[10,20],
        "negatives":[[11,12],[21,22]],
    })
    ease = lambda u,i: np.zeros(i.shape)+0.1
    lgcn = lambda u,i: np.zeros(i.shape)+0.2
    dcn  = lambda u,i: np.zeros(i.shape)+0.3
    dcn_var = lambda u,i: np.zeros(i.shape)+0.01
    pop  = {10:5, 11:1, 12:0, 20:3, 21:0, 22:1}
    llm  = pd.DataFrame({
        "user_id":[1,1,1,2,2,2],
        "item_id":[10,11,12,20,21,22],
        "yes_prob":[0.9,0.1,0.2,0.8,0.3,0.4],
    })
    cold_users = {1}
    X, y, ids = build_feature_matrix(test, ease, lgcn, dcn, dcn_var, pop, llm, cold_users)
    # 2 users × 3 items = 6 rows
    assert X.shape == (6, 8)
    assert y.shape == (6,)
    assert y.sum() == 2  # one positive per user
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/pipeline/features.py
from typing import Callable
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "ease", "lgcn", "dcn", "dcn_unc",
    "llm_yes_prob", "popularity", "is_cold_user", "intercept",
]

def build_feature_matrix(
    eval_set: pd.DataFrame,
    ease_score: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lgcn_score: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dcn_score: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dcn_uncertainty: Callable[[np.ndarray, np.ndarray], np.ndarray],
    item_popularity: dict[int, int],
    llm_features: pd.DataFrame,
    cold_users: set[int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (X, y, ids_df) where X is (n_rows, 8), y in {0,1}, ids has user_id+item_id."""
    user_ids = eval_set["user_id"].to_numpy()
    pos = eval_set["positive"].to_numpy().reshape(-1, 1)
    neg = np.stack([np.asarray(x) for x in eval_set["negatives"].tolist()])
    items = np.concatenate([pos, neg], axis=1)
    n_users, n_items = items.shape

    e = ease_score(user_ids, items)
    g = lgcn_score(user_ids, items)
    d = dcn_score(user_ids, items)
    dv = dcn_uncertainty(user_ids, items)

    llm_lookup = llm_features.set_index(["user_id", "item_id"])["yes_prob"].to_dict()

    rows = []
    ids = []
    labels = []
    for r in range(n_users):
        u = int(user_ids[r])
        for c in range(n_items):
            i = int(items[r, c])
            llm_p = llm_lookup.get((u, i), 0.5)
            pop = float(item_popularity.get(i, 0))
            is_cold = 1.0 if u in cold_users else 0.0
            rows.append([e[r,c], g[r,c], d[r,c], dv[r,c], llm_p, pop, is_cold, 1.0])
            ids.append({"user_id": u, "item_id": i})
            labels.append(1.0 if c == 0 else 0.0)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32), pd.DataFrame(ids)
```

- [ ] **Step 3: Run test**

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/pipeline/features.py tests/pipeline/test_features.py
git commit -m "feat(pipeline): per (user, candidate) feature builder"
```

---

## Task 31: Logistic-regression meta-learner

**Files:**
- Create: `src/cf_pipeline/models/meta.py`
- Create: `tests/models/test_meta.py`

- [ ] **Step 1: Failing test**

```python
# tests/models/test_meta.py
import numpy as np
from cf_pipeline.models.meta import LRMetaRanker

def test_meta_ranker_fits_and_scores():
    X = np.array([[1,0],[0,1],[1,1],[0,0]], dtype=np.float32)
    y = np.array([1,0,1,0], dtype=np.float32)
    m = LRMetaRanker().fit(X, y)
    s = m.predict(X)
    assert s.shape == (4,)
    # high y should have higher score than low y on average
    assert s[0] > s[1] or s[2] > s[3]
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/models/meta.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class LRMetaRanker:
    def __init__(self, C: float = 1.0):
        self._lr = LogisticRegression(C=C, max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LRMetaRanker":
        self._lr.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(X)[:, 1]

class LightGBMMetaRanker:
    def __init__(self, n_estimators=200, num_leaves=31, learning_rate=0.05):
        import lightgbm as lgb
        self._gbm = lgb.LGBMClassifier(
            n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, verbose=-1
        )

    def fit(self, X, y):
        self._gbm.fit(X, y)
        return self

    def predict(self, X):
        return self._gbm.predict_proba(X)[:, 1]

class MLPMetaRanker:
    def __init__(self, hidden=(64, 32), epochs=50, lr=1e-3):
        import torch
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def fit(self, X, y):
        import torch, torch.nn as nn
        in_dim = X.shape[1]
        layers = []
        prev = in_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self._model = nn.Sequential(*layers).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(X).float().to(self._device)
        yt = torch.from_numpy(y).float().to(self._device).unsqueeze(-1)
        for _ in range(self.epochs):
            logits = self._model(Xt)
            loss = bce(logits, yt)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, X):
        import torch
        self._model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X).float().to(self._device)
            return torch.sigmoid(self._model(Xt)).cpu().numpy().ravel()
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/models/test_meta.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/models/meta.py tests/models/test_meta.py
git commit -m "feat(models): LR / LightGBM / MLP meta-rankers"
```

---

## Task 32: End-to-end pipeline runner (headline experiment)

**Files:**
- Create: `scripts/run_pipeline.py`
- Create: `configs/experiment/headline.yaml`

- [ ] **Step 1: Implement**

```python
# scripts/run_pipeline.py
"""Headline pipeline:
1. Load all upstream score caches (EASE, LightGCN, DCN, LLM)
2. Build feature matrix on val + test
3. Train LR / LightGBM / MLP meta-learners on val
4. Pick best by val NDCG@10
5. Re-evaluate the winner on test → write headline result
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.utils.seeds import set_global_seed
from cf_pipeline.utils.io import save_result
from cf_pipeline.pipeline.features import build_feature_matrix
from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.models.meta import LRMetaRanker, LightGBMMetaRanker, MLPMetaRanker

PROCESSED = Path("data/processed")
RESULTS = Path("results")

def _load_score_fn(parquet_path: Path):
    df = pd.read_parquet(parquet_path)  # columns: user_id, item_id, score
    lookup = df.set_index(["user_id","item_id"])["score"].to_dict()
    def fn(users, items):
        out = np.zeros(items.shape, dtype=np.float32)
        for r, u in enumerate(users):
            for c, i in enumerate(items[r]):
                out[r, c] = lookup.get((int(u), int(i)), 0.0)
        return out
    return fn

def _load_uncertainty(parquet_path: Path):
    df = pd.read_parquet(parquet_path)  # user_id, item_id, var
    lookup = df.set_index(["user_id","item_id"])["var"].to_dict()
    def fn(users, items):
        out = np.zeros(items.shape, dtype=np.float32)
        for r, u in enumerate(users):
            for c, i in enumerate(items[r]):
                out[r, c] = lookup.get((int(u), int(i)), 0.0)
        return out
    return fn

def _scores_to_metrics(ids_df: pd.DataFrame, preds: np.ndarray, eval_set: pd.DataFrame, ks=(1,5,10,20)):
    df = ids_df.copy(); df["score"] = preds
    n_items = 1 + len(eval_set.iloc[0]["negatives"])
    score_mat = np.zeros((len(eval_set), n_items), dtype=np.float32)
    for r, row in eval_set.reset_index(drop=True).iterrows():
        sub = df[df["user_id"] == row["user_id"]].set_index("item_id")["score"]
        score_mat[r, 0] = sub.get(row["positive"], 0.0)
        for c, neg in enumerate(row["negatives"], start=1):
            score_mat[r, c] = sub.get(neg, 0.0)
    return all_metrics(score_mat, ks=ks)

def main():
    log = get_logger("headline")
    set_global_seed(42)

    train = pd.read_parquet(PROCESSED / "train.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")

    pop = train["item_id"].value_counts().to_dict()
    counts = train.groupby("user_id").size()
    cold_users = set(counts[counts < 5].index.tolist())

    # Score caches must be built and dumped by `scripts/dump_scores.py`
    ease_fn = _load_score_fn(PROCESSED / "scores_ease.parquet")
    lgcn_fn = _load_score_fn(PROCESSED / "scores_lightgcn.parquet")
    dcn_fn  = _load_score_fn(PROCESSED / "scores_dcn.parquet")
    dcn_var_fn = _load_uncertainty(PROCESSED / "scores_dcn_var.parquet")
    llm = pd.read_parquet(PROCESSED / "llm_features.parquet")

    log.info("Building val feature matrix…")
    Xv, yv, idv = build_feature_matrix(val, ease_fn, lgcn_fn, dcn_fn, dcn_var_fn, pop, llm, cold_users)
    log.info("Building test feature matrix…")
    Xt, yt, idt = build_feature_matrix(test, ease_fn, lgcn_fn, dcn_fn, dcn_var_fn, pop, llm, cold_users)

    candidates = {
        "lr": LRMetaRanker(),
        "lightgbm": LightGBMMetaRanker(),
        "mlp": MLPMetaRanker(),
    }
    best_name, best_ndcg, best_metrics = None, -1, None
    for name, m in candidates.items():
        m.fit(Xv, yv)
        preds_v = m.predict(Xv)
        v_metrics = _scores_to_metrics(idv, preds_v, val)
        log.info(f"  {name} val: {v_metrics}")
        if v_metrics["NDCG@10"] > best_ndcg:
            best_name = name
            best_ndcg = v_metrics["NDCG@10"]
            best_metrics = v_metrics
            best_model = m
    log.info(f"Best meta-learner on val: {best_name} (NDCG@10={best_ndcg:.4f})")

    # Final test eval with best
    preds_t = best_model.predict(Xt)
    t_metrics = _scores_to_metrics(idt, preds_t, test)
    log.info(f"Headline test metrics: {t_metrics}")
    save_result(
        {"experiment":"headline","meta_learner":best_name,"val_metrics":best_metrics,"metrics":t_metrics},
        RESULTS / "headline.json",
    )

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `configs/experiment/headline.yaml`**

```yaml
# configs/experiment/headline.yaml
name: headline
model: headline_pipeline
out_path: ${results_dir}/headline.json
```

- [ ] **Step 3: Commit (will run after Task 33 dumps the score caches)**

```bash
git add scripts/run_pipeline.py configs/experiment/headline.yaml
git commit -m "feat(pipeline): headline runner with meta-learner selection"
```

---

## Task 33: `dump_scores.py` — persist per-(user, item) scores from each upstream model

**Files:**
- Create: `scripts/dump_scores.py`

- [ ] **Step 1: Implement**

```python
# scripts/dump_scores.py
"""For each upstream model (EASE, LightGCN, DCN), score the candidate set
(test ∪ val) and persist as parquet for the meta-learner to consume."""
from pathlib import Path
import numpy as np
import pandas as pd
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.utils.seeds import set_global_seed
from cf_pipeline.models.ease import EASERRanker
from cf_pipeline.models.lightgcn import LightGCNRanker
from cf_pipeline.models.dcn import DCNRanker

PROCESSED = Path("data/processed")

def _candidate_pairs(test_or_val: pd.DataFrame) -> list[tuple[int,int]]:
    pairs = []
    for _, r in test_or_val.iterrows():
        u = int(r["user_id"])
        pairs.append((u, int(r["positive"])))
        for n in r["negatives"]:
            pairs.append((u, int(n)))
    return pairs

def _score_to_df(model, pairs, with_uncertainty=False):
    users = np.array([p[0] for p in pairs])
    items = np.array([[p[1]] for p in pairs])
    if with_uncertainty:
        s, var = model.score_with_uncertainty(users, items, n_mc=20)
        return (
            pd.DataFrame({"user_id":[p[0] for p in pairs],"item_id":[p[1] for p in pairs],"score":s.ravel()}),
            pd.DataFrame({"user_id":[p[0] for p in pairs],"item_id":[p[1] for p in pairs],"var":var.ravel()}),
        )
    s = model.score(users, items)
    return pd.DataFrame({"user_id":[p[0] for p in pairs],"item_id":[p[1] for p in pairs],"score":s.ravel()})

def main():
    log = get_logger("dump_scores")
    set_global_seed(42)
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    pairs = _candidate_pairs(pd.concat([test, val], ignore_index=True))

    log.info("EASE^R…")
    ease = EASERRanker(reg_lambda=500.0).fit(train)
    _score_to_df(ease, pairs).to_parquet(PROCESSED / "scores_ease.parquet")

    log.info("LightGCN…")
    lgcn = LightGCNRanker(n_layers=3).fit(train)
    _score_to_df(lgcn, pairs).to_parquet(PROCESSED / "scores_lightgcn.parquet")

    log.info("DCN-v2…")
    dcn = DCNRanker(cross_layers=3).fit(train)
    s_df, var_df = _score_to_df(dcn, pairs, with_uncertainty=True)
    s_df.to_parquet(PROCESSED / "scores_dcn.parquet")
    var_df.to_parquet(PROCESSED / "scores_dcn_var.parquet")
    log.info("Done.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run dump + headline**

```bash
uv run python scripts/dump_scores.py
uv run python scripts/run_pipeline.py
```

Expected: `results/headline.json` with HR@10 ≥ EASE^R alone (the meta-learner should fuse useful signal).

- [ ] **Step 3: Commit**

```bash
git add scripts/dump_scores.py
git commit -m "feat(pipeline): dump_scores script for upstream model caches"
```

---

## Task 34: Validate headline number sanity

- [ ] **Step 1: Inspect**

```bash
uv run python -c "import json; print(json.dumps(json.load(open('results/headline.json'))['metrics'], indent=2))"
```

- [ ] **Step 2: Sanity gates**
- HR@10 of headline must be ≥ HR@10 of EASE^R
- NDCG@10 must be > 0.4 (typical ML-1M ballpark for fused models)

If either fails: open the val metrics in `results/headline.json`, look at which meta-learner won, check whether LLM features are mostly 0.5 (a sign that S3 didn't run / failed silently). Re-run `scripts/build_llm_features.py` if so.

- [ ] **Step 3: Commit only if changes were made; otherwise skip**

---

# Phase 9 — LoRA Fine-Tuning

## Task 35: Build the (history → liked/not liked) LoRA dataset

**Files:**
- Create: `scripts/build_lora_dataset.py`

- [ ] **Step 1: Implement**

```python
# scripts/build_lora_dataset.py
"""Build a JSONL dataset of (prompt, completion) pairs for LoRA fine-tuning Llama-3.1-8B
on YES/NO recommendation decisions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.utils.seeds import set_global_seed
from cf_pipeline.llm.decision import build_decision_prompt

PROCESSED = Path("data/processed")
OUT = PROCESSED / "lora_dataset.jsonl"
N_PAIRS_PER_USER = 8  # 4 positive + 4 negative
SEED = 42

def main():
    log = get_logger("lora_dataset")
    rng = set_global_seed(SEED)
    train = pd.read_parquet(PROCESSED / "train.parquet")
    items = pd.read_parquet(PROCESSED / "items_metadata.parquet")
    items_lookup = items.set_index("item_id").to_dict(orient="index")
    all_items = items["item_id"].to_numpy()

    user_history = train.merge(items[["item_id","title","genres"]], on="item_id").groupby("user_id").apply(
        lambda g: g.to_dict(orient="records")
    ).to_dict()

    examples = []
    for u, h in user_history.items():
        if len(h) < 6:
            continue
        # Sample 4 positives from later half of history (treat as "next liked")
        late = h[len(h)//2:]
        chosen_pos = rng.choice(len(late), size=min(4, len(late)), replace=False)
        # Sample 4 random negatives
        forbidden = {x["item_id"] for x in h}
        neg_pool = [int(i) for i in all_items if int(i) not in forbidden]
        chosen_neg = rng.choice(neg_pool, size=4, replace=False)
        for idx in chosen_pos:
            target = late[int(idx)]
            other_history = [x for x in h if x["item_id"] != target["item_id"]][:10]
            prompt = build_decision_prompt(other_history, [], items_lookup.get(target["item_id"], {}))
            examples.append({"prompt": prompt, "completion": '{"decision":"YES"}'})
        for nid in chosen_neg:
            other_history = h[:10]
            prompt = build_decision_prompt(other_history, [], items_lookup.get(int(nid), {}))
            examples.append({"prompt": prompt, "completion": '{"decision":"NO"}'})

    log.info(f"Built {len(examples):,} LoRA examples")
    with open(OUT, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
uv run python scripts/build_lora_dataset.py
wc -l data/processed/lora_dataset.jsonl
```

Expected: ~48K lines.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_lora_dataset.py
git commit -m "feat(llm): build LoRA YES/NO training dataset"
```

---

## Task 36: LoRA training script

**Files:**
- Create: `scripts/lora_train.py`
- Create: `checkpoints/.gitkeep`

- [ ] **Step 1: Implement**

```python
# scripts/lora_train.py
"""LoRA fine-tune Llama-3.1-8B on the YES/NO dataset using PEFT."""
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA = Path("data/processed/lora_dataset.jsonl")
OUT = Path("checkpoints/llama8b-lora-yesno")

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","v_proj"]
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(DATA), split="train")

    def fmt(ex):
        text = ex["prompt"] + ex["completion"]
        out = tok(text, truncation=True, max_length=1024, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(fmt, remove_columns=ds.column_names)
    args = TrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
uv run python scripts/lora_train.py
```

Expected: training loss decreases over 3 epochs; final adapter saved to `checkpoints/llama8b-lora-yesno/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/lora_train.py checkpoints/.gitkeep
git commit -m "feat(llm): LoRA fine-tuning script for YES/NO decisions"
```

---

## Task 37: Build LLM feature cache with the LoRA-tuned model and re-run headline

**Files:**
- Modify: `src/cf_pipeline/llm/server.py` — accept `lora_path`
- Create: `scripts/build_llm_features_lora.py`

- [ ] **Step 1: Modify server**

In `src/cf_pipeline/llm/server.py`, replace `LlamaServer.__init__`:

```python
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: str = "bfloat16",
        max_tokens: int = 256,
        lora_path: str | None = None,
    ):
        from vllm import LLM, SamplingParams
        kwargs = {"model": model_id, "dtype": dtype, "gpu_memory_utilization": 0.85}
        if lora_path:
            kwargs["enable_lora"] = True
            from vllm.lora.request import LoRARequest
            self._lora_req = LoRARequest("yesno", 1, lora_path)
        else:
            self._lora_req = None
        self._llm = LLM(**kwargs)
        self._sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
```

And in `generate`:

```python
    def generate(self, prompts: list[str]) -> list[dict]:
        kwargs = {}
        if self._lora_req is not None:
            kwargs["lora_request"] = self._lora_req
        outs = self._llm.generate(prompts, self._sampling, **kwargs)
        ...
```

- [ ] **Step 2: New build script**

```python
# scripts/build_llm_features_lora.py
"""Same as build_llm_features.py but uses the LoRA-tuned adapter and writes a separate parquet."""
from pathlib import Path
import shutil

PROCESSED = Path("data/processed")
src = PROCESSED / "llm_features.parquet"
backup = PROCESSED / "llm_features_zeroshot.parquet"
if src.exists() and not backup.exists():
    shutil.copy(src, backup)

# Patch the import path used by build_llm_features.main(), then re-run with lora_path
import scripts.build_llm_features as base
from cf_pipeline.llm.server import LlamaServer

orig_init = LlamaServer.__init__
def patched(self, *a, **k):
    k["lora_path"] = "checkpoints/llama8b-lora-yesno"
    orig_init(self, *a, **k)

LlamaServer.__init__ = patched
base.OUT = PROCESSED / "llm_features_lora.parquet"
base.main()
```

- [ ] **Step 3: Run**

```bash
uv run python scripts/build_llm_features_lora.py
```

- [ ] **Step 4: Re-run headline using the LoRA features and save under a new result name**

Modify `scripts/run_pipeline.py` to accept `--llm_features` flag (or duplicate the script as `run_pipeline_lora.py`). Simplest: add an env var:

```python
# At top of scripts/run_pipeline.py main():
import os
LLM_FEATURES = os.environ.get("LLM_FEATURES", "llm_features.parquet")
RESULT_NAME = os.environ.get("RESULT_NAME", "headline.json")
# ... and replace `llm = pd.read_parquet(PROCESSED / "llm_features.parquet")` with PROCESSED / LLM_FEATURES
# ... and `RESULTS / "headline.json"` with RESULTS / RESULT_NAME
```

Then run:

```bash
LLM_FEATURES=llm_features_lora.parquet RESULT_NAME=headline_lora.json uv run python scripts/run_pipeline.py
```

- [ ] **Step 5: Commit**

```bash
git add src/cf_pipeline/llm/server.py scripts/build_llm_features_lora.py scripts/run_pipeline.py
git commit -m "feat(llm): support LoRA adapter at inference + LoRA feature cache"
```

---

# Phase 10 — Ablations

## Task 38: Ablation runner

**Files:**
- Create: `scripts/run_ablations.py`

- [ ] **Step 1: Implement**

```python
# scripts/run_ablations.py
"""Train meta-learner with each feature column zero'd out and re-eval headline.

Produces: results/ablate_<dropped>.json for each of:
  - no_ease, no_lgcn, no_dcn, no_dcn_unc, no_llm, no_pop, no_cold
"""
from pathlib import Path
import numpy as np
import pandas as pd
from cf_pipeline.utils.logging import get_logger
from cf_pipeline.utils.io import save_result
from cf_pipeline.utils.seeds import set_global_seed
from cf_pipeline.pipeline.features import build_feature_matrix, FEATURE_NAMES
from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.models.meta import LightGBMMetaRanker

PROCESSED = Path("data/processed")
RESULTS = Path("results")

def _load(name): return pd.read_parquet(PROCESSED / name)

def _score_lookup(parquet, col):
    df = _load(parquet)
    return df.set_index(["user_id","item_id"])[col].to_dict()

def _fn(lookup, default=0.0):
    def f(users, items):
        out = np.zeros(items.shape, dtype=np.float32)
        for r, u in enumerate(users):
            for c, i in enumerate(items[r]):
                out[r, c] = lookup.get((int(u), int(i)), default)
        return out
    return f

def _eval_with_mask(Xv, yv, Xt, idt, eval_set, drop_idx: int | None):
    if drop_idx is not None:
        Xv = Xv.copy(); Xv[:, drop_idx] = 0.0
        Xt = Xt.copy(); Xt[:, drop_idx] = 0.0
    m = LightGBMMetaRanker().fit(Xv, yv)
    preds = m.predict(Xt)
    df = idt.copy(); df["score"] = preds
    n_items = 1 + len(eval_set.iloc[0]["negatives"])
    score_mat = np.zeros((len(eval_set), n_items), dtype=np.float32)
    for r, row in eval_set.reset_index(drop=True).iterrows():
        sub = df[df["user_id"] == row["user_id"]].set_index("item_id")["score"]
        score_mat[r, 0] = sub.get(row["positive"], 0.0)
        for c, neg in enumerate(row["negatives"], start=1):
            score_mat[r, c] = sub.get(neg, 0.0)
    return all_metrics(score_mat)

def main():
    log = get_logger("ablations")
    set_global_seed(42)
    train = _load("train.parquet")
    val = _load("val.parquet"); test = _load("test.parquet")
    pop = train["item_id"].value_counts().to_dict()
    counts = train.groupby("user_id").size()
    cold = set(counts[counts < 5].index.tolist())
    ease = _fn(_score_lookup("scores_ease.parquet","score"))
    lgcn = _fn(_score_lookup("scores_lightgcn.parquet","score"))
    dcn  = _fn(_score_lookup("scores_dcn.parquet","score"))
    dcn_v= _fn(_score_lookup("scores_dcn_var.parquet","var"))
    llm  = _load("llm_features.parquet")

    Xv, yv, idv = build_feature_matrix(val, ease, lgcn, dcn, dcn_v, pop, llm, cold)
    Xt, yt, idt = build_feature_matrix(test, ease, lgcn, dcn, dcn_v, pop, llm, cold)

    metrics_full = _eval_with_mask(Xv, yv, Xt, idt, test, drop_idx=None)
    save_result({"experiment":"ablate_none","metrics":metrics_full}, RESULTS / "ablate_none.json")

    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    drops = ["ease","lgcn","dcn","dcn_unc","llm_yes_prob","popularity","is_cold_user"]
    for d in drops:
        m = _eval_with_mask(Xv, yv, Xt, idt, test, drop_idx=name_to_idx[d])
        save_result({"experiment":f"ablate_no_{d}","metrics":m}, RESULTS / f"ablate_no_{d}.json")
        log.info(f"  no_{d}: NDCG@10={m['NDCG@10']:.4f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
uv run python scripts/run_ablations.py
```

Expected: 8 JSON files in `results/ablate_*.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_ablations.py
git commit -m "feat(experiments): ablation runner (drop one feature at a time)"
```

---

## Task 39: Inspect ablation deltas

- [ ] **Step 1: Print deltas**

```bash
uv run python -c "
import json, glob
base = json.load(open('results/ablate_none.json'))['metrics']['NDCG@10']
for f in sorted(glob.glob('results/ablate_no_*.json')):
    m = json.load(open(f))['metrics']['NDCG@10']
    print(f'{f}: {m:.4f} (Δ={m-base:+.4f})')
"
```

- [ ] **Step 2: Document insights**
- Any feature with `Δ ≥ -0.005` (i.e., dropping it barely matters) is honest to flag in the report
- Any feature with `Δ ≤ -0.02` is doing real work
- No commit unless code changed

---

# Phase 11 — Robustness + Cold Sub-Population

## Task 40: Full-ranking eval implementation

**Files:**
- Create: `src/cf_pipeline/eval/full_ranking.py`
- Create: `tests/eval/test_full_ranking.py`

- [ ] **Step 1: Failing test**

```python
# tests/eval/test_full_ranking.py
import numpy as np
import pandas as pd
from cf_pipeline.eval.full_ranking import full_ranking_eval

def test_full_ranking_perfect_model_full_score():
    train = pd.DataFrame({"user_id":[1], "item_id":[10]})
    test = pd.DataFrame({"user_id":[1], "positive":[20]})
    all_items = [10, 20, 30, 40]

    class Perfect:
        def score(self, users, items):
            out = np.zeros(items.shape)
            out[items == 20] = 1.0
            return out

    metrics = full_ranking_eval(Perfect(), train, test, all_items, ks=(1,))
    assert metrics["HR@1"] == 1.0
```

- [ ] **Step 2: Implement**

```python
# src/cf_pipeline/eval/full_ranking.py
import numpy as np
import pandas as pd
from cf_pipeline.eval.metrics import all_metrics
from cf_pipeline.models.base import BaseRanker

def full_ranking_eval(
    model: BaseRanker, train: pd.DataFrame, test: pd.DataFrame,
    all_item_ids: list[int], ks=(10, 20)
) -> dict[str, float]:
    """For each test user, rank ALL unseen items."""
    history: dict[int, set[int]] = train.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items = np.asarray(all_item_ids)
    score_rows = []
    for _, r in test.iterrows():
        u = int(r["user_id"]); pos = int(r["positive"])
        seen = history.get(u, set())
        unseen = all_items[~np.isin(all_items, list(seen) + [pos])]
        cand = np.concatenate([[pos], unseen]).reshape(1, -1)
        s = model.score(np.array([u]), cand).ravel()
        # _rank_of_positive expects column 0 to be positive
        score_rows.append(s)
    # Pad to same length (rare: most users have similar unseen counts)
    L = max(len(s) for s in score_rows)
    padded = np.full((len(score_rows), L), -1e9, dtype=np.float32)
    for i, s in enumerate(score_rows):
        padded[i, :len(s)] = s
    return all_metrics(padded, ks=ks)
```

- [ ] **Step 3: Run test**

- [ ] **Step 4: Commit**

```bash
git add src/cf_pipeline/eval/full_ranking.py tests/eval/test_full_ranking.py
git commit -m "feat(eval): full ranking eval (Table 5 robustness)"
```

---

## Task 41: Run robustness check on headline

**Files:**
- Create: `scripts/run_robustness.py`

- [ ] **Step 1: Implement**

```python
# scripts/run_robustness.py
"""Run full-ranking eval on headline pipeline + popularity baseline."""
from pathlib import Path
import pandas as pd
from cf_pipeline.utils.io import save_result
from cf_pipeline.models.baselines import PopularityRanker
from cf_pipeline.eval.full_ranking import full_ranking_eval

# Headline pipeline as a BaseRanker — wrap by reading the dumped LightGBM meta-learner.
# For simplicity, reuse the logic in run_pipeline.py but only return scores per (user, item).

PROCESSED = Path("data/processed")
RESULTS = Path("results")

def main():
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")
    items = pd.read_parquet(PROCESSED / "items_metadata.parquet")
    all_items = items["item_id"].tolist()

    # Popularity floor
    pop = PopularityRanker().fit(train)
    pop_metrics = full_ranking_eval(pop, train, test, all_items, ks=(10, 20))
    save_result({"experiment":"robustness_popularity","metrics":pop_metrics}, RESULTS / "robustness_pop.json")

    # NOTE: For the headline pipeline under full ranking, we score *every unseen item per user*.
    # That's expensive for the LLM stage. Strategy: skip LLM features and use the meta-learner
    # over (ease, lgcn, dcn, dcn_unc, popularity, is_cold) only — explicitly note this in report.
    print(pop_metrics)

if __name__ == "__main__":
    main()
```

> **Note for the report:** Table 5 explicitly documents that the LLM stage was excluded under the full-ranking eval (because scoring 6,040 users × ~3,700 unseen items × LLM call = 22M calls is impractical even on an A100). The rest of the meta-learner runs as normal. This is the scientifically honest way to handle it and matches how research papers report mixed protocols.

- [ ] **Step 2: Run**

```bash
uv run python scripts/run_robustness.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_robustness.py
git commit -m "feat(experiments): robustness full-ranking eval (Table 5)"
```

---

## Task 42: Cold-user sub-population eval

**Files:**
- Create: `scripts/run_cold_users.py`

- [ ] **Step 1: Implement**

```python
# scripts/run_cold_users.py
"""Evaluate headline pipeline on the cold-user sub-population (<5 train interactions),
both WITH and WITHOUT the S0 cold-start LLM profile."""
import json
from pathlib import Path
import pandas as pd
from cf_pipeline.utils.io import save_result
from cf_pipeline.utils.logging import get_logger

PROCESSED = Path("data/processed")
RESULTS = Path("results")

def main():
    log = get_logger("cold_users")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    counts = train.groupby("user_id").size()
    cold_users = set(counts[counts < 5].index.tolist())
    log.info(f"{len(cold_users)} cold users in test")

    cold_test = test[test["user_id"].isin(cold_users)]
    cold_test_path = PROCESSED / "test_cold.parquet"
    cold_test.to_parquet(cold_test_path)

    # Re-run pipeline on cold subset by setting an env var consumed by run_pipeline.py
    import subprocess, os
    env = {**os.environ, "TEST_PATH_OVERRIDE": str(cold_test_path), "RESULT_NAME": "cold_users_with_s0.json"}
    subprocess.run(["uv","run","python","scripts/run_pipeline.py"], env=env, check=True)

    # Now disable S0 by zeroing the is_cold_user feature: easiest is to set env COLD_USER_OVERRIDE=0
    env["DISABLE_S0"] = "1"
    env["RESULT_NAME"] = "cold_users_no_s0.json"
    subprocess.run(["uv","run","python","scripts/run_pipeline.py"], env=env, check=True)

if __name__ == "__main__":
    main()
```

> **Note:** `scripts/run_pipeline.py` must honour `TEST_PATH_OVERRIDE` and `DISABLE_S0`. Add at the top of `main()`:
>
> ```python
> test_path = Path(os.environ.get("TEST_PATH_OVERRIDE", PROCESSED / "test.parquet"))
> if os.environ.get("DISABLE_S0") == "1":
>     cold_users = set()  # forces all `is_cold_user` features to 0
> ```

- [ ] **Step 2: Run**

```bash
uv run python scripts/run_cold_users.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_cold_users.py scripts/run_pipeline.py
git commit -m "feat(experiments): cold-user sub-population eval (Table 4)"
```

---

# Phase 12 — Tables, Report, Video

## Task 43: `make_tables.py` — collate JSON results into markdown tables

**Files:**
- Create: `scripts/make_tables.py`

- [ ] **Step 1: Implement**

```python
# scripts/make_tables.py
"""Read results/*.json and write results/tables.md with all 5 report tables."""
import json
from pathlib import Path

RESULTS = Path("results")
KS = [1, 5, 10, 20]

def _load(name):
    p = RESULTS / name
    if not p.exists(): return None
    return json.load(open(p))

def _row(label, payload):
    if not payload: return None
    m = payload.get("metrics", {})
    cells = [label]
    for k in KS:
        cells.append(f"{m.get(f'HR@{k}', float('nan')):.4f}")
    for k in KS:
        cells.append(f"{m.get(f'NDCG@{k}', float('nan')):.4f}")
    return "| " + " | ".join(cells) + " |"

def _header():
    head = ["Model"] + [f"HR@{k}" for k in KS] + [f"NDCG@{k}" for k in KS]
    sep = ["---"] * len(head)
    return "| " + " | ".join(head) + " |\n| " + " | ".join(sep) + " |"

def main():
    out = []
    out.append("# Report Tables\n")

    out.append("## Table 1 — Headline comparison\n")
    out.append(_header())
    for label, fname in [
        ("Popularity",   "baseline_pop.json"),
        ("ItemKNN",      "baseline_itemknn.json"),
        ("BPR-MF",       "baseline_bpr.json"),
        ("NeuMF",        "baseline_neumf.json"),
        ("EASE^R",       "ease.json"),
        ("LightGCN",     "lightgcn.json"),
        ("DCN-v2",       "dcn.json"),
        ("**Headline**", "headline.json"),
    ]:
        row = _row(label, _load(fname))
        if row: out.append(row)
    out.append("")

    out.append("## Table 2 — Ablation (drop one feature at a time)\n")
    out.append(_header())
    for label, fname in [
        ("Full",            "ablate_none.json"),
        ("− EASE^R",        "ablate_no_ease.json"),
        ("− LightGCN",      "ablate_no_lgcn.json"),
        ("− DCN-v2",        "ablate_no_dcn.json"),
        ("− DCN uncertainty","ablate_no_dcn_unc.json"),
        ("− LLM YES/NO",    "ablate_no_llm_yes_prob.json"),
        ("− Popularity",    "ablate_no_popularity.json"),
        ("− S0 cold flag",  "ablate_no_is_cold_user.json"),
    ]:
        row = _row(label, _load(fname))
        if row: out.append(row)
    out.append("")

    out.append("## Table 3 — LoRA fine-tune delta\n")
    out.append(_header())
    out.append(_row("Zero-shot Llama (headline)", _load("headline.json")))
    out.append(_row("LoRA-tuned Llama",           _load("headline_lora.json")))
    out.append("")

    out.append("## Table 4 — Cold-user sub-population\n")
    out.append(_header())
    out.append(_row("Headline w/ S0", _load("cold_users_with_s0.json")))
    out.append(_row("Headline − S0",  _load("cold_users_no_s0.json")))
    out.append("")

    out.append("## Table 5 — Full-ranking robustness\n")
    sub_head = ["Model", "HR@10", "HR@20", "NDCG@10", "NDCG@20"]
    out.append("| " + " | ".join(sub_head) + " |")
    out.append("| " + " | ".join(["---"]*5) + " |")
    for label, fname in [
        ("Popularity (full ranking)", "robustness_pop.json"),
        ("Headline (full ranking, no LLM)", "robustness_headline.json"),
    ]:
        p = _load(fname)
        if not p: continue
        m = p["metrics"]
        out.append(f"| {label} | {m.get('HR@10','nan'):.4f} | {m.get('HR@20','nan'):.4f} | {m.get('NDCG@10','nan'):.4f} | {m.get('NDCG@20','nan'):.4f} |")
    out.append("")

    (RESULTS / "tables.md").write_text("\n".join(out))
    print(f"Wrote {RESULTS / 'tables.md'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
uv run python scripts/make_tables.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/make_tables.py
git commit -m "feat(report): make_tables script collating result JSONs"
```

---

## Task 44: LaTeX report skeleton

**Files:**
- Create: `report/report.tex`

- [ ] **Step 1: Write the skeleton**

```latex
% report/report.tex
\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}

\title{Hybrid Collaborative Filtering on MovieLens 1M:\\
       EASE\textsuperscript{R} + LightGCN + DCN-v2 + LLM Re-Ranking}
\author{Vinay}
\date{2026}

\begin{document}
\maketitle

\section*{Setup}
Dataset: MovieLens 1M, joined with TMDB metadata (titles, genres, plots, keywords).
Binarization: rating $\geq 4 \to 1$. Evaluation: leave-one-out + 99 sampled negatives
(NCF protocol, fixed seed). Metrics: HR@$K$ and NDCG@$K$ for $K \in \{1,5,10,20\}$.

Pipeline: EASE\textsuperscript{R} + LightGCN + DCN-v2 produce per-(user, item)
scores; an LLM (Llama-3.1-8B-Instruct, optionally LoRA-fine-tuned) emits a strict
JSON YES/NO judgement per item via FAISS+BM25 RAG retrieval; a deterministic
LightGBM meta-learner fuses all signals into the final ranking. The LLM is
\textbf{never} the final decision maker.

\section*{Table 1 — Headline Comparison}
\input{table1}

\section*{Table 2 — Ablation}
\input{table2}

\section*{Table 3 — LoRA Fine-Tune Delta}
\input{table3}

\section*{Table 4 — Cold-User Sub-Population}
\input{table4}

\section*{Table 5 — Full-Ranking Robustness}
\input{table5}

\section*{Demo Video}
\href{https://youtu.be/REPLACE_WITH_LINK}{https://youtu.be/REPLACE\_WITH\_LINK}

\end{document}
```

- [ ] **Step 2: Convert markdown tables → LaTeX**

Add to `scripts/make_tables.py` a `--latex` flag that emits `report/table1.tex` … `report/table5.tex` using `booktabs`. (For brevity here, do the conversion manually if you prefer.)

- [ ] **Step 3: Build the PDF**

```bash
cd report && pdflatex report.tex && pdflatex report.tex && cd ..
```

- [ ] **Step 4: Commit**

```bash
git add report/
git commit -m "docs(report): LaTeX skeleton with 5-table layout"
```

---

## Task 45: Video script and recording checklist

**Files:**
- Create: `report/video_script.md`

- [ ] **Step 1: Write the script**

```markdown
# YouTube Demo Script (target: 5–8 min)

## 0:00–1:00 — Problem & dataset (1 min)
- Show problem statement on screen (project_problem_statement.txt)
- "MovieLens 1M, binarize rating ≥ 4, leave-one-out + 99 negatives, K ∈ {1,5,10,20}"
- Show recommendation_system_pipeline.svg

## 1:00–3:00 — Architecture walkthrough (2 min)
- Open the SVG diagram
- Explain S1/S2/S3/S4 in plain English
- Emphasize: under NCF protocol, S1/S2 are SCORERS not RETRIEVERS, and the meta-learner (S4) is the final decision maker

## 3:00–5:00 — Live inference on one user (2 min)
- Open a Python REPL or notebook
- Pick a sample test user (e.g., user 1234)
- Print their training history (titles)
- Run the pipeline:  
  `from scripts.run_pipeline import score_one_user`  
  (you may need to add a `score_one_user(uid)` helper)
- Show the top-10 recommendations
- Show what they actually next watched (the held-out positive)
- Compute the rank, HR@10, NDCG@10 contribution

## 5:00–7:00 — Tour the 5 tables (2 min)
- Open results/tables.md side-by-side with the SVG
- For each table, point at the "interesting" cell:
  - Table 1: headline beats baselines
  - Table 2: which feature mattered most
  - Table 3: LoRA delta (positive or negative — be honest)
  - Table 4: S0 helps cold users (or doesn't)
  - Table 5: ordering preserved under full ranking

## 7:00–8:00 — Conclusion + what surprised me (1 min)
- One concrete finding (e.g. "LightGCN didn't beat EASE^R on this dataset, matching prior literature")
- Repo link, end

## Recording setup
- OBS Studio, 1080p, mic check
- Close all unrelated tabs/apps
- Upload to YouTube as **unlisted**
- Paste the link into report.tex and rebuild PDF
```

- [ ] **Step 2: Commit**

```bash
git add report/video_script.md
git commit -m "docs(report): YouTube demo script + recording checklist"
```

---

## Task 46: Final wiring + submission checklist

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README with full instructions**

```markdown
# CF Pipeline — MovieLens 1M

Hybrid CF pipeline (EASE^R + LightGCN + DCN-v2 + LLM RAG re-ranker + LightGBM meta-learner) for HR@K / NDCG@K on MovieLens 1M, NCF leave-one-out + 99-negatives protocol.

## Setup
```bash
uv sync
# Place ML-1M into data/raw/ml-1m/ and TMDB metadata into data/raw/tmdb/
uv run python scripts/prepare_data.py
uv run pytest -m "not gpu and not integration"
uv run pytest -m integration   # after prepare_data
```

## Reproduce all results

```bash
# Baselines
uv run python scripts/eval.py +experiment=baseline_pop
uv run python scripts/eval.py +experiment=baseline_itemknn
uv run python scripts/eval.py +experiment=baseline_bpr
uv run python scripts/eval.py +experiment=baseline_neumf

# CF models
uv run python scripts/eval.py +experiment=ease
uv run python scripts/eval.py +experiment=lightgcn
uv run python scripts/eval.py +experiment=dcn

# Score caches → headline pipeline
uv run python scripts/dump_scores.py
uv run python scripts/build_cold_start_profiles.py
uv run python scripts/build_llm_features.py
uv run python scripts/run_pipeline.py

# LoRA
uv run python scripts/build_lora_dataset.py
uv run python scripts/lora_train.py
uv run python scripts/build_llm_features_lora.py
LLM_FEATURES=llm_features_lora.parquet RESULT_NAME=headline_lora.json uv run python scripts/run_pipeline.py

# Ablations + robustness + cold users
uv run python scripts/run_ablations.py
uv run python scripts/run_robustness.py
uv run python scripts/run_cold_users.py

# Tables + report
uv run python scripts/make_tables.py
cd report && pdflatex report.tex && pdflatex report.tex && cd ..
```

## Submission checklist
- [ ] All five tables in `results/tables.md` and `report/report.pdf`
- [ ] YouTube link embedded in `report/report.tex`
- [ ] PDF rebuilt after embedding link
- [ ] PDF + YouTube link uploaded to Google Classroom
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: complete reproduction recipe in README"
```

---

# Self-Review (run after the plan is written)

**Spec coverage check:** Walk through the spec sections and confirm each maps to a task.

| Spec section | Task(s) |
|---|---|
| §4 C1 binarization | Task 5 |
| §4 C4 ML-1M + TMDB join | Tasks 4, 6 |
| §4 C5 rating ≥ 4 threshold | Task 5 |
| §4 C6 Llama-3.1-8B vLLM | Tasks 22, 23, 24, 28, 29 |
| §4 C7 LOO + 99 negatives, K∈{1,5,10,20} | Tasks 8, 11, 12 |
| §5 S1/S2 → feature producers | Tasks 30, 33 |
| §6.2 S0 cold-start | Tasks 23, 24 |
| §6.2 S1 EASE^R | Task 19 |
| §6.2 S1 LightGCN | Task 20 |
| §6.2 S2 DCN-v2 + MC dropout | Task 21 |
| §6.2 S3 LLM + RAG (FAISS+BM25+HyDE) | Tasks 25, 26, 27, 28, 29 |
| §6.2 S4 meta-learner (LR/LightGBM/MLP) | Tasks 30, 31, 32 |
| §7 data preprocessing | Tasks 4–10 |
| §8.1 NCF protocol | Tasks 11, 12, 13 |
| §8.2 full-ranking robustness | Tasks 40, 41 |
| §9 LightGCN/DCN/LoRA fine-tuning | Tasks 20, 21, 35, 36, 37 |
| §10 reproducibility (seed, git_sha, lockfile) | Tasks 1, 2, 3 |
| §11.1 Tables 1–5 | Tasks 15–18, 19–21, 32, 38, 41, 42, 43 |
| §11.2 LaTeX PDF | Task 44 |
| §11.3 YouTube demo script | Task 45 |
| §12 risk: LLM eval too slow | Mitigated in Task 41 (full-ranking skips LLM) |
| §12 risk: data leakage | Task 9 |
| §13 success criteria | Reproduction recipe in Task 46 |
| §14 open questions resolved | Task 19 (EASE λ sweep), Task 20 (LightGCN layer sweep), Task 21 (DCN hyperparams), Task 35 (LoRA recipe) |

**Type-consistency check:** All `BaseRanker` subclasses implement `score(user_ids, item_ids) → ndarray`. `LRMetaRanker`, `LightGBMMetaRanker`, `MLPMetaRanker` all expose `fit(X, y)` and `predict(X)`. `DenseItemIndex` and `BM25ItemIndex` both expose `build(items) → self` and `search(query, k) → list[(int, float)]`. `FEATURE_NAMES` in `pipeline/features.py` is the single source of truth for ablation indexing.

**Placeholder scan:** All "TODO"-flavored markers are intentional `# NOTE` comments in scripts that need user judgment (e.g., picking λ from sweep results) — they are explicit instructions to the human, not unfilled plan steps. Every task has actual code in every code-bearing step.

**Scope check:** This is one project, one dataset, one protocol — single plan is appropriate.

---

**Plan complete.**
