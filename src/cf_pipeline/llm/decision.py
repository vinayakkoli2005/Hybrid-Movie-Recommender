from __future__ import annotations

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

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def build_decision_prompt(
    user_history: list[dict],
    retrieved: list[dict],
    candidate: dict,
) -> str:
    h = "\n".join(
        f"- {x.get('title', '?')} ({x.get('genres', '?')})" for x in user_history[:10]
    )
    r = "\n".join(
        f"- {x.get('title', '?')} ({x.get('genres', '?')})" for x in retrieved[:5]
    )
    return DECISION_TEMPLATE.format(
        history_block=h,
        retrieved_block=r,
        candidate_title=candidate.get("title", "?"),
        candidate_genres=candidate.get("genres", "?"),
        candidate_overview=candidate.get("overview", ""),
    )


def parse_decision_response(text: str, logprobs) -> dict:
    """Returns {decision: 'YES'|'NO', yes_prob: float}."""
    decision = "NO"
    m = _JSON_RE.search(text or "")
    if m:
        try:
            d = json.loads(m.group(0))
            if str(d.get("decision", "")).upper() == "YES":
                decision = "YES"
        except json.JSONDecodeError:
            pass

    # logprob extraction is best-effort (our server returns None for now)
    yes_prob = None
    if logprobs:
        for tok_lp in logprobs:
            if not tok_lp:
                continue
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
