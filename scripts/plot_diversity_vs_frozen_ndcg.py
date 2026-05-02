"""Plot Diversity vs NDCG using frozen-negative metrics from results/tuned_pipeline.json.
Saves to results/figures/diversity_vs_ndcg_frozen.png
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

summary_path = FIGS / "summary.json"
tuned_path = ROOT / "results" / "tuned_pipeline.json"

if not summary_path.exists():
    raise SystemExit("Run scripts/metrics_postproc.py first to produce summary.json")

summary = json.loads(summary_path.read_text())
if tuned_path.exists():
    tuned = json.loads(tuned_path.read_text())
    ks = summary.get("ks", [1,5,10,20])
    avg_div = summary.get("avg_diversity", [])
    # extract NDCG@k from tuned metrics if present
    tuned_metrics = tuned.get("metrics", {})
    ndcgs = [tuned_metrics.get(f"NDCG@{k}", None) for k in ks]
else:
    raise SystemExit("tuned_pipeline.json not found; cannot plot frozen NDCG")

fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(ks, avg_div, marker='o', color='#ff79c6')
ax1.set_xlabel('K (top-K)')
ax1.set_ylabel('Avg Diversity (unique genres)', color='#ff79c6')
ax1.tick_params(axis='y', labelcolor='#ff79c6')

ax2 = ax1.twinx()
ax2.plot(ks, ndcgs, marker='s', color='#8be9fd')
ax2.set_ylabel('Frozen-Negative NDCG', color='#8be9fd')
ax2.tick_params(axis='y', labelcolor='#8be9fd')

fig.tight_layout()
out = FIGS / 'diversity_vs_ndcg_frozen.png'
fig.savefig(out, dpi=200)
print('Wrote', out)
