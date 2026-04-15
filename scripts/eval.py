"""Hydra evaluation entry-point for the CF pipeline.

Usage
-----
# Evaluate popularity baseline on test split (default)
    python scripts/eval.py

# Switch experiment
    python scripts/eval.py experiment=baseline_itemknn

# Evaluate on validation split
    python scripts/eval.py eval.split=val

# Full override example
    python scripts/eval.py experiment=ease eval.split=val seed=0

Config hierarchy (see configs/)
    config.yaml
      ├─ data:       ml1m
      ├─ eval:       ncf_protocol
      └─ experiment: baseline_pop  (default; override with experiment=<name>)

Result JSON is written to cfg.experiment.out_path (default: results/<name>.json).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from cf_pipeline.eval.protocol import run_and_save_experiment
from cf_pipeline.utils.seeds import set_global_seed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory — maps cfg.experiment.model → instantiated BaseRanker
# ---------------------------------------------------------------------------

def _build_model(cfg: DictConfig, train: pd.DataFrame):
    """Instantiate and fit the model named in cfg.experiment.model.

    Each branch reads its own hyperparameters from cfg.experiment so they can
    be overridden from the CLI without touching YAML files.
    """
    model_name: str = cfg.experiment.model

    if model_name == "popularity":
        from cf_pipeline.models.baselines import PopularityRanker
        return PopularityRanker().fit(train)

    if model_name == "itemknn":
        from cf_pipeline.models.baselines import ItemKNNRanker
        k   = int(cfg.experiment.get("k_neighbors", 20))
        shr = float(cfg.experiment.get("shrinkage",   10.0))
        log.info("ItemKNN — k_neighbors=%d  shrinkage=%.1f", k, shr)
        return ItemKNNRanker(k_neighbors=k, shrinkage=shr).fit(train)

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Implement it in src/cf_pipeline/models/ and register it here."
    )


# ---------------------------------------------------------------------------
# Hydra main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    log.info("=== CF Pipeline Evaluation ===")
    log.info("Experiment : %s", cfg.experiment.name)
    log.info("Model      : %s", cfg.experiment.model)
    log.info("Split      : %s", cfg.eval.split)
    log.info("K values   : %s", list(cfg.eval.ks))
    log.info("Output     : %s", cfg.experiment.out_path)

    # ------------------------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------------------------
    processed_dir = Path(cfg.data.processed_dir)

    train_path = processed_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run `python scripts/prepare_data.py` first."
        )

    train = pd.read_parquet(train_path)
    log.info("Loaded train: %d rows, %d users, %d items",
             len(train),
             train["user_id"].nunique(),
             train["item_id"].nunique())

    # Choose val or test split
    split = cfg.eval.split
    eval_path = processed_dir / f"eval_negatives_{split}.json"
    if not eval_path.exists():
        # Fallback: prepare_data.py may have written a single eval_negatives.json
        fallback = processed_dir / "eval_negatives.json"
        if split == "test" and fallback.exists():
            eval_path = fallback
            log.warning("Using legacy eval_negatives.json for test split.")
        else:
            raise FileNotFoundError(
                f"Eval negatives not found at {eval_path}. "
                "Run `python scripts/prepare_data.py` first."
            )

    with open(eval_path) as f:
        raw = json.load(f)

    # Support two formats:
    #   - legacy nested: {"test": [...], "val": [...]}   (from prepare_data.py)
    #   - flat list:     [{"user_id": ..., ...}, ...]
    if isinstance(raw, dict) and split in raw:
        eval_records = raw[split]
    elif isinstance(raw, list):
        eval_records = raw
    else:
        raise ValueError(
            f"eval_negatives JSON must be either a list or a dict with key '{split}'. "
            f"Found keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
        )
    eval_set = pd.DataFrame(eval_records)
    log.info("Loaded eval set (%s): %d users × 100 candidates", split, len(eval_set))

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = _build_model(cfg, train)
    log.info("Model built: %s", type(model).__name__)

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    ks = tuple(cfg.eval.ks)
    out_path = Path(cfg.experiment.out_path)

    payload = run_and_save_experiment(
        model=model,
        eval_set=eval_set,
        experiment_name=cfg.experiment.name,
        out_path=out_path,
        ks=ks,
    )

    # ------------------------------------------------------------------
    # 4. Pretty-print results
    # ------------------------------------------------------------------
    metrics = payload["metrics"]
    header = f"\n{'='*52}\n  Results — {cfg.experiment.name} ({split})\n{'='*52}"
    log.info(header)
    print(header)

    col_w = 12
    ks_sorted = sorted({int(k.split("@")[1]) for k in metrics})
    header_row = f"{'Metric':<10}" + "".join(f"{'@'+str(k):>{col_w}}" for k in ks_sorted)
    print(header_row)
    print("-" * len(header_row))

    for metric_prefix in ("HR", "NDCG"):
        row = f"{metric_prefix:<10}"
        for k in ks_sorted:
            key = f"{metric_prefix}@{k}"
            val = metrics.get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("=" * 52)
    print(f"Results saved → {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
