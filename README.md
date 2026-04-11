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
