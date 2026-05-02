"""Generate pipeline flowchart using matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 22))
fig.patch.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, title, subtitle="", color="#bd93f9", bg="#1e1e2e"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.12", linewidth=1.5,
                         edgecolor=color, facecolor=bg, zorder=3)
    ax.add_patch(box)
    if subtitle:
        ax.text(x, y + 0.13, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color, zorder=4)
        ax.text(x, y - 0.2, subtitle, ha="center", va="center",
                fontsize=7.5, color="#aaaaaa", zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color="#555577"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=2)

def stage_label(ax, x, y, text, color="#444466"):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=8, color=color, style="italic", zorder=4)

# ── Stage backgrounds ─────────────────────────────────────────────────────────
stages = [
    (0.3, 21.3, 13.5, 1.2, "#0f0f22", "Stage 1 · Data Preparation"),
    (0.3, 18.9, 13.5, 2.2, "#0f1a0f", "Stage 2 · Base CF Models"),
    (0.3, 16.9, 13.5, 1.2, "#0f1a12", "Stage 3 · LLM Features  (Optional)"),
    (0.3, 15.0, 13.5, 1.2, "#1a0f1a", "Stage 4 · Feature Engineering"),
    (0.3, 13.1, 13.5, 1.2, "#1a1a0f", "Stage 5 · Hyperparameter Optimization"),
    (0.3, 11.2, 13.5, 1.2, "#1a100f", "Stage 6 · Final Training"),
    (0.3,  9.3, 13.5, 1.2, "#0f1a1a", "Stage 7 · Evaluation"),
    (0.3,  7.3, 13.5, 1.2, "#1a0f1a", "Stage 8 · Gradio UI"),
]
for (x, y, w, h, bg, label) in stages:
    rect = FancyBboxPatch((x, y - h), w, h,
                          boxstyle="round,pad=0.1", linewidth=1,
                          edgecolor="#2a2a3a", facecolor=bg, zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.2, y - 0.12, label, ha="left", va="top",
            fontsize=7.5, color="#444466", style="italic", zorder=2)

# ── Stage 1: Data ─────────────────────────────────────────────────────────────
draw_box(ax, 3,   21.0, 3.2, 0.6, "ML-1M Dataset", "1M ratings · 6K users · 4K movies", "#8be9fd")
draw_box(ax, 7,   21.0, 3.2, 0.6, "Leave-One-Out Split", "train / val / test.parquet", "#8be9fd")
draw_box(ax, 11,  21.0, 3.2, 0.6, "99 Negatives / User", "for evaluation protocol", "#8be9fd")
draw_arrow(ax, 4.6, 21.0, 5.4, 21.0)
draw_arrow(ax, 8.6, 21.0, 9.4, 21.0)

# ── Stage 2: CF Models ────────────────────────────────────────────────────────
models = ["Pop", "ItemKNN", "EASE", "BPR", "LightGCN", "DCN", "NeuMF", "SASRec"]
xs = [1.1, 2.9, 4.7, 6.5, 7.0, 8.8, 10.6, 12.4]
for name, x in zip(models, xs):
    draw_box(ax, x, 19.4, 1.6, 0.5, name, "", "#ff79c6")
    draw_arrow(ax, x, 19.15, x, 18.55)

draw_box(ax, 7, 18.2, 10.5, 0.55, "cf_scores_val/test.parquet", "(user, item, 8 CF scores)", "#ffb86c")

# arrow from data to CF models
draw_arrow(ax, 11, 20.7, 7, 19.68, "#444466")

# ── Stage 3: LLM ──────────────────────────────────────────────────────────────
draw_box(ax, 4,  16.5, 3.5, 0.6, "LoRA Fine-tuned LLM", "binary yes/no per (user, item)", "#50fa7b")
draw_box(ax, 10, 16.5, 3.5, 0.6, "llm_features.parquet", "yes_prob column", "#50fa7b")
draw_arrow(ax, 5.75, 16.5, 8.25, 16.5)

# ── Stage 4: Features ─────────────────────────────────────────────────────────
draw_box(ax, 2.8, 14.6, 3.2, 0.6, "Rank-Normalize CF Scores", "per-user percentile [0,1]", "#bd93f9")
draw_box(ax, 7.0, 14.6, 3.2, 0.6, "User & Item Features", "n_interactions · avg_pop · pop_rank", "#bd93f9")
draw_box(ax, 11.2,14.6, 2.8, 0.6, "LLM yes_prob", "fused signal", "#bd93f9")
draw_box(ax, 7.0, 13.55,4.5, 0.6, "13-dim Feature Vector / (user, item)", "", "#ffb86c")

draw_arrow(ax, 2.8, 14.3, 5.2, 13.8)
draw_arrow(ax, 7.0, 14.3, 7.0, 13.8)
draw_arrow(ax, 11.2,14.3, 8.8, 13.8)

# arrows into stage 4
draw_arrow(ax, 7.0, 17.93, 7.0, 14.93, "#444466")
draw_arrow(ax, 10, 16.2, 11.2, 14.93, "#444466")

# ── Stage 5: HPO ──────────────────────────────────────────────────────────────
draw_box(ax, 4.5, 12.65, 3.8, 0.6, "Optuna TPE Search", "80 trials · SQLite · no retesting", "#f1fa8c")
draw_box(ax, 10,  12.65, 4.5, 0.6, "Best Params", "lr=0.08 · leaves=98 · depth=3 · ff=0.62", "#f1fa8c")
draw_arrow(ax, 6.4, 12.65, 7.75, 12.65)
draw_arrow(ax, 7.0, 13.25, 4.5, 12.95, "#444466")

# ── Stage 6: Training ─────────────────────────────────────────────────────────
draw_box(ax, 4.5, 10.75, 4.0, 0.6, "LightGBM LambdaRank", "optimizes NDCG directly", "#ffb86c")
draw_box(ax, 10,  10.75, 3.5, 0.6, "meta_lgbm_tuned.pkl", "saved checkpoint", "#ffb86c")
draw_arrow(ax, 6.5, 10.75, 8.25, 10.75)
draw_arrow(ax, 10,  12.35, 7.5,  11.05, "#444466")
draw_arrow(ax, 7.0, 13.25, 4.5,  11.05, "#444466")

# ── Stage 7: Eval ─────────────────────────────────────────────────────────────
draw_box(ax, 4.5,  8.85, 4.5, 0.6, "Metrics @ 1/5/10/20", "HR · NDCG · MAP · MAR · Novelty", "#8be9fd")
draw_box(ax, 10.5, 8.85, 3.5, 0.6, "Results", "HR@10≈0.79  NDCG@10≈0.62", "#8be9fd")
draw_arrow(ax, 6.75, 8.85, 8.75, 8.85)
draw_arrow(ax, 10,  10.45, 4.5,  9.15, "#444466")

# ── Stage 8: UI ───────────────────────────────────────────────────────────────
draw_box(ax, 7, 7.4, 9.5, 0.65,
         "Gradio App  ·  port 7860",
         "User Selector  ·  Top-20 Recs  ·  Metric Cards  ·  Movie Prediction",
         "#ff79c6", "#2a1a2e")
draw_arrow(ax, 10,  10.45, 7, 7.73, "#444466")
draw_arrow(ax, 4.5, 8.55,  7, 7.73, "#444466")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7, 21.85, "Hybrid Movie Recommender — Full Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="#bd93f9")
ax.text(7, 21.6, "ML-1M  ·  8 CF Models  ·  LambdaRank Meta-Learner  ·  Gradio UI",
        ha="center", va="center", fontsize=9, color="#888888")

plt.tight_layout()
plt.savefig("pipeline_flowchart.png", dpi=150, bbox_inches="tight",
            facecolor="#0d0d1a")
print("Saved: pipeline_flowchart.png")
