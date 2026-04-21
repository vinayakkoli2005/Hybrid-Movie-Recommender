from __future__ import annotations

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
