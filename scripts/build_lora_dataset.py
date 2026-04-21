"""Build LoRA fine-tuning dataset from ML-1M train split.

Strategy per user:
  - Sample min(POS_PER_USER, |history|) positives: leave one item out, rest = context → YES
  - For each positive, sample 1 negative (random non-interacted item) → NO

Output: data/processed/lora_train.jsonl
Each line: {"prompt": <str>, "response": '{"decision": "YES"}' | '{"decision": "NO"}'}
Target: ~50-60K balanced pairs.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from cf_pipeline.llm.decision import build_decision_prompt
from cf_pipeline.utils.logging import get_logger

PROCESSED = Path("data/processed")
OUT = PROCESSED / "lora_train.jsonl"

POS_PER_USER = 5   # positive samples per user
SEED = 42
MIN_HISTORY = 3    # skip users with fewer interactions (not enough context)


def main() -> None:
    log = get_logger("build_lora_dataset")
    random.seed(SEED)

    train = pd.read_parquet(PROCESSED / "train.parquet")
    items = pd.read_parquet(PROCESSED / "items_metadata.parquet")
    items_lookup = items.set_index("item_id").to_dict(orient="index")
    all_item_ids = set(items["item_id"].tolist())

    # Build per-user history ordered by timestamp
    user_items: dict[int, list[int]] = (
        train.sort_values("timestamp")
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    log.info("Building dataset from %d users…", len(user_items))

    records: list[dict] = []

    for uid, history in user_items.items():
        if len(history) < MIN_HISTORY:
            continue

        # Pool of negatives: items the user has NOT interacted with
        seen = set(history)
        neg_pool = list(all_item_ids - seen)
        if not neg_pool:
            continue

        # Sample positives (leave-one-out style)
        n_pos = min(POS_PER_USER, len(history) - 1)
        sampled_pos_indices = random.sample(range(1, len(history)), n_pos)

        for pos_idx in sampled_pos_indices:
            positive_item = history[pos_idx]
            # Context: all history except the positive (use up to 10 most recent)
            context = [h for i, h in enumerate(history) if i != pos_idx][-10:]

            # Build history and retrieved blocks from context items
            context_meta = [
                items_lookup.get(iid, {"title": "?", "genres": "?", "overview": ""})
                for iid in context
            ]
            # Use last 5 context items as "retrieved" (simple proxy for FAISS retrieval)
            retrieved = context_meta[-5:]
            cand_pos_meta = items_lookup.get(
                positive_item, {"title": "?", "genres": "?", "overview": ""}
            )

            # Positive pair
            prompt_pos = build_decision_prompt(context_meta, retrieved, cand_pos_meta)
            records.append({"prompt": prompt_pos, "response": '{"decision": "YES"}'})

            # Negative pair — random non-interacted item
            neg_item = random.choice(neg_pool)
            cand_neg_meta = items_lookup.get(
                neg_item, {"title": "?", "genres": "?", "overview": ""}
            )
            prompt_neg = build_decision_prompt(context_meta, retrieved, cand_neg_meta)
            records.append({"prompt": prompt_neg, "response": '{"decision": "NO"}'})

    # Shuffle before writing
    random.shuffle(records)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    yes_count = sum(1 for r in records if "YES" in r["response"])
    no_count = len(records) - yes_count
    log.info(
        "Done — %d total pairs (YES=%d, NO=%d) → %s",
        len(records), yes_count, no_count, OUT,
    )


if __name__ == "__main__":
    main()
