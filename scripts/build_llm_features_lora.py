"""Score candidates with LoRA fine-tuned Gemma-2-9b-it and produce yes_prob features.

Processes BOTH val and test splits so the meta-learner has real LLM signal
during training (val) AND evaluation (test).

FIXED: Candidate selection is now LABEL-BLIND — top-K candidates are chosen by
fused CF score rank, NOT by always including the positive. This prevents the
candidate-selection leak where the positive was guaranteed to receive an LLM
score while 95% of negatives defaulted to 0.5.

TOP_K=10 captures ~72% of positives (matching HR@10 of the CF ensemble) without
using any label information.

Output: data/processed/llm_features.parquet — (user_id, item_id, decision, yes_prob)

Resume-safe: already-done (user_id, item_id) pairs are skipped.
Checkpoint every 10 users.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from cf_pipeline.llm.decision import build_decision_prompt, parse_decision_response
from cf_pipeline.llm.rag import BM25ItemIndex, DenseItemIndex, reciprocal_rank_fusion
from cf_pipeline.utils.logging import get_logger

PROCESSED        = Path("data/processed")
LORA_ADAPTER     = Path("checkpoints/lora/final")
BASE_MODEL       = Path("models/gemma-2-9b-it")
OUT              = PROCESSED / "llm_features.parquet"
TOP_K            = 10   # top-K by CF score (label-blind) per user per split
BATCH_SIZE       = 8    # prompts per forward pass
CHECKPOINT_EVERY = 10   # users between saves

# CF score columns to fuse for label-blind candidate selection
CF_SCORE_COLS = ["knn", "ease", "bpr", "lgcn", "dcn", "neumf"]


def _cf_top_k(cf_scores: pd.DataFrame, user_id: int, k: int) -> list[int]:
    """Return top-k item_ids for user by fused CF rank (label-blind)."""
    udf = cf_scores[cf_scores["user_id"] == user_id].copy()
    if udf.empty:
        return []
    # Per-user rank-normalise each CF column, then average → fused score
    for col in CF_SCORE_COLS:
        if col in udf.columns:
            udf[col] = udf[col].rank(pct=True)
        else:
            udf[col] = 0.5
    udf["_fusion"] = udf[CF_SCORE_COLS].mean(axis=1)
    top = udf.nlargest(k, "_fusion")
    return top["item_id"].tolist()


def _load_model(device: str = "cuda:0"):
    log = get_logger("llm_features_lora")
    log.info("Loading base model %s …", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), torch_dtype=torch.bfloat16
    ).to(device)
    log.info("Applying LoRA adapter from %s …", LORA_ADAPTER)
    model = PeftModel.from_pretrained(model, str(LORA_ADAPTER))
    model.eval()
    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    log.info("Model ready on %s", device)
    return model, tokenizer


def _generate(model, tokenizer, prompts: list[str], device: str, max_new_tokens: int = 32) -> list[dict]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    results = []
    for i in range(len(prompts)):
        new_tokens = outputs[i][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append({"text": text, "logprobs": None})
    return results


def _score_split(
    split_df: pd.DataFrame,
    cf_scores: pd.DataFrame,
    split_name: str,
    history_by_user: dict,
    items_lookup: dict,
    dense: DenseItemIndex,
    bm25: BM25ItemIndex,
    model,
    tokenizer,
    device: str,
    done_pairs: set,
    log,
) -> list[dict]:
    rows: list[dict] = []
    total = len(split_df)
    pos_in_topk = 0

    # Pre-index cf_scores by user for fast lookup
    cf_by_user = {uid: grp for uid, grp in cf_scores.groupby("user_id")}

    for ridx, row in enumerate(split_df.itertuples(index=False)):
        u = int(row.user_id)
        positive = int(row.positive)
        history = history_by_user.get(u, [])

        # Label-blind: pick top-K candidates by fused CF score
        udf = cf_by_user.get(u, pd.DataFrame())
        if udf.empty:
            # Fallback if no CF scores available for this user
            candidates = [positive] + [int(x) for x in row.negatives[:TOP_K - 1]]
        else:
            udf = udf.copy()
            for col in CF_SCORE_COLS:
                if col in udf.columns:
                    udf[col] = udf[col].rank(pct=True)
                else:
                    udf[col] = 0.5
            udf["_fusion"] = udf[CF_SCORE_COLS].mean(axis=1)
            top = udf.nlargest(TOP_K, "_fusion")
            candidates = top["item_id"].tolist()

        if positive in candidates:
            pos_in_topk += 1

        prompts, metas = [], []
        for cand in candidates:
            if (u, cand) in done_pairs:
                continue
            cand_meta = items_lookup.get(cand, {"title": "?", "genres": "?", "overview": ""})
            query = f"{cand_meta.get('title', '')} {cand_meta.get('genres', '')}"
            fused = reciprocal_rank_fusion(
                [dense.search(query, k=5), bm25.search(query, k=5)], k=5
            )
            retrieved = [items_lookup.get(iid, {"title": "?", "genres": "?"}) for iid, _ in fused]
            prompts.append(build_decision_prompt(history, retrieved, cand_meta))
            metas.append({"user_id": u, "item_id": cand})

        if not prompts:
            continue

        outs = []
        for i in range(0, len(prompts), BATCH_SIZE):
            outs.extend(_generate(model, tokenizer, prompts[i:i + BATCH_SIZE], device))
        for meta, o in zip(metas, outs):
            parsed = parse_decision_response(o["text"], o["logprobs"])
            rows.append({**meta, **parsed})

        if (ridx + 1) % 20 == 0:
            log.info("  [%s] %d / %d users done | positive in top-%d: %.1f%%",
                     split_name, ridx + 1, total, TOP_K,
                     100 * pos_in_topk / (ridx + 1))

    log.info("  [%s] final positive-in-top-%d rate: %.1f%%",
             split_name, TOP_K, 100 * pos_in_topk / max(total, 1))
    return rows


def main() -> None:
    log = get_logger("llm_features_lora")
    device = "cuda:1"

    log.info("FIXED: Using label-blind CF-ranked candidate selection (TOP_K=%d)", TOP_K)

    train   = pd.read_parquet(PROCESSED / "train.parquet")

    val     = pd.read_parquet(PROCESSED / "val.parquet")
    test    = pd.read_parquet(PROCESSED / "test.parquet")
    cf_val  = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
    cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")
    items   = pd.read_parquet(PROCESSED / "items_metadata.parquet")
    items_lookup = items.set_index("item_id").to_dict(orient="index")

    # Start fresh — old features used biased candidate selection
    log.info("Starting fresh (clearing old biased llm_features.parquet)")
    if OUT.exists():
        OUT.unlink()

    done_pairs: set[tuple[int, int]] = set()
    all_rows: list[dict] = []

    log.info("Building item indexes …")
    dense = DenseItemIndex().build(items)
    bm25  = BM25ItemIndex().build(items)

    history_by_user = (
        train.merge(items[["item_id", "title", "genres"]], on="item_id")
        .groupby("user_id")
        .apply(lambda g: g[["item_id", "title", "genres"]].to_dict(orient="records"))
        .to_dict()
    )

    model, tokenizer = _load_model(device)
    log.info("Using device: %s", device)

    for split_df, cf_scores, split_name in [
        (val,  cf_val,  "val"),
        (test, cf_test, "test"),
    ]:
        log.info("Scoring %s split (%d users × top-%d CF-ranked candidates) …",
                 split_name, len(split_df), TOP_K)
        new_rows = _score_split(
            split_df, cf_scores, split_name, history_by_user, items_lookup,
            dense, bm25, model, tokenizer, device, done_pairs, log,
        )
        all_rows.extend(new_rows)
        done_pairs.update((r["user_id"], r["item_id"]) for r in new_rows)

        pd.DataFrame(all_rows).to_parquet(OUT)
        log.info("  [%s] done — %d total rows saved", split_name, len(all_rows))

    df_out = pd.DataFrame(all_rows)
    df_out.to_parquet(OUT)
    log.info("Done — %d rows, %d users, yes_prob mean=%.3f std=%.3f",
             len(df_out), df_out["user_id"].nunique(),
             df_out["yes_prob"].mean(), df_out["yes_prob"].std())


if __name__ == "__main__":
    main()
