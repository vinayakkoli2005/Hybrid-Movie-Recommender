import json
from pathlib import Path

import pandas as pd
import pytest

from cf_pipeline.eval.protocol import run_and_save_experiment
from tests.eval.test_protocol import _AlwaysFavorPositive, _AlwaysFavorNegative


# ── helpers ───────────────────────────────────────────────────────────────────

def _small_eval_set():
    return pd.DataFrame({
        "user_id":   [1, 2],
        "positive":  [10, 20],
        "negatives": [[11, 12, 13], [21, 22, 23]],
    })


# ── plan test ─────────────────────────────────────────────────────────────────

def test_writes_json_with_metrics(tmp_path):
    out = tmp_path / "result.json"
    run_and_save_experiment(
        model=_AlwaysFavorPositive(),
        eval_set=_small_eval_set(),
        experiment_name="fake",
        out_path=out,
    )
    with open(out) as f:
        data = json.load(f)
    assert data["experiment"] == "fake"
    assert data["metrics"]["HR@1"] == pytest.approx(1.0)
    assert "timestamp" in data


# ── extra correctness ─────────────────────────────────────────────────────────

def test_all_k_values_in_output(tmp_path):
    out = tmp_path / "r.json"
    run_and_save_experiment(
        model=_AlwaysFavorPositive(),
        eval_set=_small_eval_set(),
        experiment_name="test",
        out_path=out,
    )
    with open(out) as f:
        data = json.load(f)
    for k in (1, 5, 10, 20):
        assert f"HR@{k}"   in data["metrics"]
        assert f"NDCG@{k}" in data["metrics"]


def test_returns_payload_dict(tmp_path):
    out = tmp_path / "r.json"
    result = run_and_save_experiment(
        model=_AlwaysFavorPositive(),
        eval_set=_small_eval_set(),
        experiment_name="retval_check",
        out_path=out,
    )
    assert isinstance(result, dict)
    assert result["experiment"] == "retval_check"
    assert "metrics" in result


def test_git_sha_in_saved_file(tmp_path):
    out = tmp_path / "r.json"
    run_and_save_experiment(
        model=_AlwaysFavorPositive(),
        eval_set=_small_eval_set(),
        experiment_name="sha_check",
        out_path=out,
    )
    with open(out) as f:
        data = json.load(f)
    assert "git_sha" in data   # injected by save_result in utils/io.py


def test_worst_ranker_hr1_zero_is_saved(tmp_path):
    out = tmp_path / "r.json"
    run_and_save_experiment(
        model=_AlwaysFavorNegative(),
        eval_set=_small_eval_set(),
        experiment_name="worst",
        out_path=out,
    )
    with open(out) as f:
        data = json.load(f)
    assert data["metrics"]["HR@1"] == pytest.approx(0.0)


def test_output_file_is_valid_json(tmp_path):
    out = tmp_path / "r.json"
    run_and_save_experiment(
        model=_AlwaysFavorPositive(),
        eval_set=_small_eval_set(),
        experiment_name="json_check",
        out_path=out,
    )
    assert out.exists()
    content = out.read_text()
    parsed = json.loads(content)   # will raise if not valid JSON
    assert isinstance(parsed, dict)
