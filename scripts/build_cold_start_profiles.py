"""Compute and cache S0 LLM profiles for users with <5 train interactions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    log.info("%d cold users (<%d train interactions)", len(cold_users), THRESHOLD)

    if not cold_users:
        OUT.write_text("{}")
        log.info("No cold users — wrote empty profile file.")
        return

    history_lookup = (
        train.merge(items, on="item_id")
        .groupby("user_id")
        .apply(lambda g: g[["title", "genres"]].to_dict(orient="records"))
        .to_dict()
    )

    srv = LlamaServer()
    profiles = {}
    for i, u in enumerate(cold_users):
        prompt = build_cold_start_prompt(u, history_lookup.get(u, []))
        out = srv.generate([prompt])
        profiles[int(u)] = parse_cold_start_response(out[0]["text"])
        if (i + 1) % 10 == 0:
            log.info("Processed %d / %d cold users", i + 1, len(cold_users))

    OUT.write_text(json.dumps(profiles, indent=2))
    log.info("Wrote %d profiles to %s", len(profiles), OUT)
    srv.free()


if __name__ == "__main__":
    main()
