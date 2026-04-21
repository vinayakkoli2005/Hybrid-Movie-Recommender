"""Generate result tables (markdown + LaTeX) from saved JSON result files.

Reads all results/*.json and produces:
  results/table1_baselines.md    — Table 1: baseline comparison
  results/table1_baselines.tex   — LaTeX version
  results/table4_cold_users.md   — Table 4: cold-user sub-population
  results/table4_cold_users.tex  — LaTeX version
  results/table5_ablation.md     — Table 5: ablation study
  results/table5_ablation.tex    — LaTeX version

Re-run after LoRA + Task 29 to refresh the LLM row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cf_pipeline.utils.logging import get_logger

RESULTS = Path("results")

# Display name mapping for model keys
MODEL_NAMES = {
    "baseline_pop":     "Popularity",
    "baseline_itemknn": "ItemKNN",
    "baseline_bpr":     "BPR-MF",
    "ease":             "EASE^R",
    "dcn_smoke":        "DCN-v2",
    "lightgcn_smoke":   "LightGCN",
    "baseline_neumf":   "NeuMF",
    "hybrid_pipeline":  "Hybrid (CF only)",
    "hybrid_pipeline_with_llm": "Hybrid (CF + LLM)",  # added after Task 29
}

# CF-only per-model results from cf_scores (computed in run_pipeline.py log)
CF_MODEL_RESULTS = {
    "pop":   {"HR@5": None, "HR@10": 0.4981, "HR@20": None, "NDCG@5": None, "NDCG@10": 0.2768, "NDCG@20": None},
    "knn":   {"HR@5": None, "HR@10": 0.6810, "HR@20": None, "NDCG@5": None, "NDCG@10": 0.3992, "NDCG@20": None},
    "ease":  {"HR@5": None, "HR@10": 0.6953, "HR@20": None, "NDCG@5": None, "NDCG@10": 0.4296, "NDCG@20": None},
    "bpr":   {"HR@5": None, "HR@10": 0.6189, "HR@20": None, "NDCG@5": None, "NDCG@10": 0.3477, "NDCG@20": None},
    "lgcn":  {"HR@5": None, "HR@10": 0.4976, "HR@20": None, "NDCG@5": None, "NDCG@10": 0.2757, "NDCG@20": None},
    "dcn":   {"HR@5": None, "HR@10": 0.6378, "HR@20": None, "NDCG@5": None, "NDCG@10": 0.3591, "NDCG@20": None},
    "neumf": {"HR@5": None, "HR@10": 0.6838, "HR@20": None, "NDCG@5": None, "NDCG@10": 0.3942, "NDCG@20": None},
}


def _load_results() -> dict[str, dict]:
    """Load all JSON result files from results/."""
    data: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            obj = json.loads(path.read_text())
            name = obj.get("experiment", path.stem)
            data[name] = obj.get("metrics", obj)
        except Exception:
            pass
    return data


def _fmt(v, bold: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:.4f}"
    return f"**{s}**" if bold else s


def _fmt_tex(v, bold: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def _best_in_col(rows: list[dict], col: str) -> float | None:
    vals = [r[col] for r in rows if r.get(col) is not None]
    return max(vals) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Baselines
# ─────────────────────────────────────────────────────────────────────────────

def _build_table1_rows() -> list[dict]:
    """Assemble Table 1 rows from CF model scores + saved hybrid result."""
    results = _load_results()

    rows = []

    # Individual CF models (from run_pipeline.py log output)
    display = {
        "pop": "Popularity", "knn": "ItemKNN",   "bpr": "BPR-MF",
        "ease": "EASE^R",    "lgcn": "LightGCN", "dcn": "DCN-v2", "neumf": "NeuMF",
    }
    for key, name in display.items():
        m = CF_MODEL_RESULTS.get(key, {})
        rows.append({"model": name, **m})

    # Hybrid pipeline (from saved JSON)
    hybrid = results.get("hybrid_pipeline", {})
    if hybrid:
        rows.append({
            "model": "Hybrid (CF only)",
            "HR@5":    hybrid.get("HR@5"),
            "HR@10":   hybrid.get("HR@10"),
            "HR@20":   hybrid.get("HR@20"),
            "NDCG@5":  hybrid.get("NDCG@5"),
            "NDCG@10": hybrid.get("NDCG@10"),
            "NDCG@20": hybrid.get("NDCG@20"),
        })

    # Tuned LambdaRank (LambdaRank + Optuna + enhanced features + retrained neural)
    tuned = results.get("tuned_lambdarank_pipeline", {})
    if tuned:
        rows.append({
            "model": "Hybrid (LambdaRank + Optuna)",
            "HR@5":    tuned.get("HR@5"),
            "HR@10":   tuned.get("HR@10"),
            "HR@20":   tuned.get("HR@20"),
            "NDCG@5":  tuned.get("NDCG@5"),
            "NDCG@10": tuned.get("NDCG@10"),
            "NDCG@20": tuned.get("NDCG@20"),
        })

    # Hybrid + LLM (filled after Task 29 LoRA scoring completes)
    hybrid_llm = results.get("hybrid_pipeline_with_llm", {})
    if hybrid_llm:
        rows.append({
            "model": "Hybrid (CF + LoRA LLM)",
            "HR@5":    hybrid_llm.get("HR@5"),
            "HR@10":   hybrid_llm.get("HR@10"),
            "HR@20":   hybrid_llm.get("HR@20"),
            "NDCG@5":  hybrid_llm.get("NDCG@5"),
            "NDCG@10": hybrid_llm.get("NDCG@10"),
            "NDCG@20": hybrid_llm.get("NDCG@20"),
        })

    return rows


def _table1_markdown(rows: list[dict]) -> str:
    cols = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
    best = {c: _best_in_col(rows, c) for c in cols}

    lines = ["# Table 1: Recommendation Performance on ML-1M (NCF leave-one-out protocol)\n"]
    header = "| Model | " + " | ".join(cols) + " |"
    sep    = "|-------|" + "|".join(["-------"] * len(cols)) + "|"
    lines += [header, sep]

    for r in rows:
        sep_row = "| --- |" + "|".join(["---"] * len(cols)) + "|" if r["model"].startswith("Hybrid") and rows.index(r) == len(rows) - (2 if any("LLM" in rr["model"] for rr in rows) else 1) else None
        if sep_row:
            lines.append(sep_row)
        cells = [_fmt(r.get(c), bold=(r.get(c) == best[c])) for c in cols]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")

    lines.append("\n*Bold = best in column. — = not reported.*")
    return "\n".join(lines)


def _table1_latex(rows: list[dict]) -> str:
    cols = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
    best = {c: _best_in_col(rows, c) for c in cols}

    col_spec = "l" + "c" * len(cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Recommendation performance on ML-1M (leave-one-out, 99 negatives).}",
        r"\label{tab:results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]

    hybrid_sep_done = False
    for r in rows:
        if r["model"].startswith("Hybrid") and not hybrid_sep_done:
            lines.append(r"\midrule")
            hybrid_sep_done = True
        cells = [_fmt_tex(r.get(c), bold=(r.get(c) == best[c])) for c in cols]
        lines.append(r["model"].replace("^", r"\^{}") + " & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 5: Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _table5_markdown(ablation: dict) -> str:
    full_ndcg = ablation.get("full", {}).get("NDCG@10", 0)
    full_hr   = ablation.get("full", {}).get("HR@10", 0)

    lines = ["# Table 5: Ablation Study — Feature Contribution (NDCG@10)\n"]
    lines.append("| Dropped Feature | HR@10 | NDCG@10 | Δ NDCG@10 |")
    lines.append("|----------------|-------|---------|-----------|")
    lines.append(f"| None (full model) | {full_hr:.4f} | {full_ndcg:.4f} | — |")

    drops = [(k.replace("drop_", ""), v) for k, v in ablation.items() if k.startswith("drop_")]
    for feat, v in sorted(drops, key=lambda x: x[1]["delta_NDCG@10"]):
        delta = v["delta_NDCG@10"]
        sign  = f"{delta:+.4f}"
        lines.append(f"| {feat} | {v['HR@10']:.4f} | {v['NDCG@10']:.4f} | {sign} |")

    lines.append("\n*Negative Δ means the feature hurts when removed.*")
    return "\n".join(lines)


def _table5_latex(ablation: dict) -> str:
    full_ndcg = ablation.get("full", {}).get("NDCG@10", 0)
    full_hr   = ablation.get("full", {}).get("HR@10", 0)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: each feature removed from the meta-learner.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccr}",
        r"\toprule",
        r"Dropped Feature & HR@10 & NDCG@10 & $\Delta$ NDCG@10 \\",
        r"\midrule",
        rf"None (full) & {full_hr:.4f} & {full_ndcg:.4f} & — \\",
        r"\midrule",
    ]

    drops = [(k.replace("drop_", ""), v) for k, v in ablation.items() if k.startswith("drop_")]
    for feat, v in sorted(drops, key=lambda x: x[1]["delta_NDCG@10"]):
        delta = v["delta_NDCG@10"]
        lines.append(rf"{feat} & {v['HR@10']:.4f} & {v['NDCG@10']:.4f} & {delta:+.4f} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Cold users
# ─────────────────────────────────────────────────────────────────────────────

def _table4_markdown(cold: dict) -> str:
    lines = ["# Table 4: Performance on Cold vs. All Users\n"]
    groups = list(cold.keys())
    lines.append("| Model | " + " | ".join(f"{g} NDCG@10" for g in groups) + " |")
    lines.append("|-------|" + "|".join(["----------"] * len(groups)) + "|")

    models = list(cold[groups[0]].keys()) if groups else []
    for model in models:
        cells = [f"{cold[g].get(model, {}).get('NDCG@10', '—'):.4f}"
                 if isinstance(cold[g].get(model, {}).get('NDCG@10'), float) else "—"
                 for g in groups]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table4_latex(cold: dict) -> str:
    groups = list(cold.keys())
    models = list(cold[groups[0]].keys()) if groups else []
    col_spec = "l" + "c" * len(groups)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{NDCG@10 on cold vs.\ all users.}",
        r"\label{tab:cold}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & " + " & ".join(g.replace("_", r"\_") for g in groups) + r" \\",
        r"\midrule",
    ]
    for model in models:
        cells = []
        for g in groups:
            v = cold[g].get(model, {}).get("NDCG@10")
            cells.append(f"{v:.4f}" if isinstance(v, float) else "—")
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log = get_logger("generate_tables")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Table 1
    log.info("Generating Table 1 (baselines)…")
    rows = _build_table1_rows()
    (RESULTS / "table1_baselines.md").write_text(_table1_markdown(rows))
    (RESULTS / "table1_baselines.tex").write_text(_table1_latex(rows))
    log.info("  → table1_baselines.md / .tex")

    # Table 5 (ablation)
    ablation_path = RESULTS / "ablation.json"
    if ablation_path.exists():
        log.info("Generating Table 5 (ablation)…")
        ablation = json.loads(ablation_path.read_text())
        (RESULTS / "table5_ablation.md").write_text(_table5_markdown(ablation))
        (RESULTS / "table5_ablation.tex").write_text(_table5_latex(ablation))
        log.info("  → table5_ablation.md / .tex")
    else:
        log.info("Skipping Table 5 — run ablation_runner.py first.")

    # Table 4 (cold users)
    cold_path = RESULTS / "cold_user_table.json"
    if cold_path.exists():
        log.info("Generating Table 4 (cold users)…")
        cold = json.loads(cold_path.read_text())
        (RESULTS / "table4_cold_users.md").write_text(_table4_markdown(cold))
        (RESULTS / "table4_cold_users.tex").write_text(_table4_latex(cold))
        log.info("  → table4_cold_users.md / .tex")
    else:
        log.info("Skipping Table 4 — run cold_user_eval.py first.")

    log.info("Done.")


if __name__ == "__main__":
    main()
