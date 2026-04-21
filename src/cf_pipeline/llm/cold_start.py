from __future__ import annotations

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

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_cold_start_prompt(user_id: int, history: list[dict]) -> str:
    lines = [f"- {h.get('title', '?')} ({h.get('genres', '?')})" for h in history]
    return PROMPT_TEMPLATE.format(history_block="\n".join(lines) if lines else "(no history)")


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
