"""Interactive Movie Recommender Demo — Gradio UI."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import numpy as np
import pandas as pd

from cf_pipeline.features_enhanced import (
    ENHANCED_FEAT_COLS,
    build_enhanced_feature_matrix,
    build_stats,
)

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")

# ── Load & precompute at startup ──────────────────────────────────────────────
print("Loading data...")
train   = pd.read_parquet(PROCESSED / "train.parquet")
cf_val  = pd.read_parquet(PROCESSED / "cf_scores_val.parquet")
cf_test = pd.read_parquet(PROCESSED / "cf_scores_test.parquet")
items   = pd.read_parquet(PROCESSED / "items_metadata.parquet")

llm_feats = None
if (PROCESSED / "llm_features.parquet").exists():
    llm_feats = pd.read_parquet(PROCESSED / "llm_features.parquet")

print("Building features & predictions...")
user_stats, item_stats = build_stats(train)
cf_all = pd.concat([cf_val, cf_test], ignore_index=True).drop_duplicates(
    subset=["user_id", "item_id"]
)
df_all = build_enhanced_feature_matrix(cf_all, user_stats, item_stats, llm_feats)

with open(CHECKPOINTS / "meta_lgbm_tuned.pkl", "rb") as f:
    model = pickle.load(f)

X = df_all[ENHANCED_FEAT_COLS].to_numpy(dtype=np.float32)
df_all["pred_score"] = model.predict(X).astype(np.float32)
df_all["rank"] = (
    df_all.groupby("user_id")["pred_score"]
    .rank(ascending=False, method="average")
    .astype(float)
)
# Percentile score: 1.0 = best, 0.0 = worst (for display)
df_all["n_cands"] = df_all.groupby("user_id")["rank"].transform("count")
df_all["pct_score"] = ((df_all["n_cands"] - df_all["rank"]) / (df_all["n_cands"] - 1)).clip(0, 1)
df_all["rank"] = df_all["rank"].round().astype(int)

items_lookup   = items.set_index("item_id").to_dict(orient="index")
all_users      = sorted(train["user_id"].unique().tolist())
train_global   = {
    "item_popularity": train.groupby("item_id").size().to_dict(),
    "n_train": len(train),
}
history_by_user = (
    train.sort_values("timestamp") if "timestamp" in train.columns else train
).groupby("user_id")["item_id"].apply(list).to_dict()

print(f"Ready — {len(all_users)} users, {len(df_all)} scored pairs.")


def get_candidates_for_user(user_id: int) -> list[str]:
    udf = df_all[df_all["user_id"] == user_id].sort_values("rank")
    titles = []
    for iid in udf["item_id"].tolist():
        meta = items_lookup.get(int(iid), {})
        titles.append(meta.get("title", f"Item {iid}"))
    return titles  # sorted by rank (best first)


def get_top20(user_id: int) -> pd.DataFrame:
    udf = df_all[df_all["user_id"] == user_id].copy()
    if udf.empty:
        return pd.DataFrame(columns=["Movie", "Genres", "Year", "Score"])
    top = udf.nsmallest(20, "rank")
    rows = []
    for _, row in top.iterrows():
        meta  = items_lookup.get(int(row["item_id"]), {})
        title = meta.get("title", f"Item {int(row['item_id'])}")
        year  = title[-5:-1] if len(title) > 6 and title[-1] == ")" else "—"
        rows.append({
            "Movie":  title,
            "Genres": meta.get("genres", "?").replace("|", " · "),
            "Year":   year,
            "Score":  f"{float(row['pct_score']):.1%}",
        })
    return pd.DataFrame(rows)


def get_metrics_html(user_id: int) -> str:
    udf = df_all[df_all["user_id"] == user_id].copy()
    if udf.empty:
        return ""
    top = udf.nsmallest(20, "rank")

    # HR@20: did positive item land in top 20?
    positives = udf[udf["label"] == 1]["rank"].tolist() if "label" in udf.columns else []
    hr20 = 1.0 if positives and positives[0] <= 20 else 0.0

    # NDCG@20: single-positive protocol
    if positives:
        r = positives[0]
        ndcg20 = float(1.0 / np.log2(r + 1)) if r <= 20 else 0.0
    else:
        ndcg20 = 0.0

    # Genre Diversity: unique genres among top-20 items
    all_genres = set()
    for iid in top["item_id"].tolist():
        g = items_lookup.get(int(iid), {}).get("genres", "")
        for genre in g.split("|"):
            if genre.strip():
                all_genres.add(genre.strip())
    diversity = len(all_genres)

    # Avg novelty of top-20
    item_popularity = train_global["item_popularity"]
    n_train = train_global["n_train"]
    novelties = []
    for iid in top["item_id"].tolist():
        pop = item_popularity.get(int(iid), 1)
        novelties.append(-np.log2(pop / n_train))
    avg_novelty = float(np.mean(novelties)) if novelties else 0.0

    def metric_card(label, value, color="#bd93f9"):
        return (
            f"<div style='background:#2a2a3a;border-radius:10px;padding:12px 18px;"
            f"text-align:center;min-width:90px;flex:1'>"
            f"<div style='font-size:1.5em;font-weight:bold;color:{color}'>{value}</div>"
            f"<div style='color:#888;font-size:0.78em;margin-top:2px'>{label}</div>"
            f"</div>"
        )

    hr_color = "#50fa7b" if hr20 == 1.0 else "#ff6e6e"
    ndcg_str = f"{ndcg20:.3f}" if positives else "N/A"
    hr_str   = "✓ Hit" if hr20 == 1.0 else "✗ Miss"

    cards = (
        metric_card("NDCG@20", ndcg_str, "#bd93f9") +
        metric_card("HR@20", hr_str, hr_color) +
        metric_card("Genre Diversity", str(diversity), "#ff79c6") +
        metric_card("Avg Novelty", f"{avg_novelty:.2f}", "#8be9fd")
    )
    return f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px'>{cards}</div>"


def get_history_html(user_id: int) -> str:
    hist = history_by_user.get(user_id, [])[-15:]
    if not hist:
        return "<p style='color:#888'>No history found.</p>"
    lines = []
    for iid in reversed(hist):
        meta = items_lookup.get(iid, {})
        title  = meta.get("title", f"Item {iid}")
        genres = meta.get("genres", "").replace("|", " · ")
        lines.append(
            f"<div style='padding:6px 0; border-bottom:1px solid #2a2a3a'>"
            f"<span style='color:#e0e0ff;font-weight:500'>{title}</span><br>"
            f"<span style='color:#888;font-size:0.85em'>{genres}</span></div>"
        )
    return "".join(lines)


def on_user_select(user_id: int):
    top20        = get_top20(user_id)
    history_html = get_history_html(user_id)
    candidates   = get_candidates_for_user(user_id)
    metrics_html = get_metrics_html(user_id)
    return top20, history_html, gr.update(choices=candidates, value=candidates[0] if candidates else None), metrics_html


def on_movie_select(user_id: int, movie_title: str) -> str:
    if not movie_title:
        return ""
    # Find item_id by title
    item_id = None
    for iid, meta in items_lookup.items():
        if meta.get("title") == movie_title:
            item_id = iid
            break
    if item_id is None:
        return "<p style='color:#f88'>Movie not found.</p>"

    row = df_all[(df_all["user_id"] == user_id) & (df_all["item_id"] == item_id)]
    meta = items_lookup.get(item_id, {})
    genres   = meta.get("genres", "?").replace("|", " · ")
    overview = meta.get("overview", "")[:250]
    year     = meta.get("release_date", "?")

    if row.empty:
        return f"""
        <div style='background:#1e1e2e;border-radius:12px;padding:20px;color:#ccc'>
          <h3 style='color:#f0a0a0'>⚠️ Not in scored candidate set</h3>
          <p><b style='color:#e0e0ff'>{movie_title}</b></p>
          <p style='color:#888'>{genres}</p>
          <p>This movie was not among the top candidates generated by the CF models for this user.
          The system considers it unlikely to be relevant.</p>
        </div>"""

    rank      = int(row["rank"].values[0])
    n_cands   = int(row["n_cands"].values[0])
    pct_score = float(row["pct_score"].values[0])
    label     = int(row["label"].values[0]) if "label" in row.columns else None

    if rank <= 5:
        verdict = "🟢 Highly Recommended"
        vcolor  = "#50fa7b"
    elif rank <= 20:
        verdict = "🟡 Likely to Enjoy"
        vcolor  = "#f1fa8c"
    else:
        verdict = "🔴 Unlikely Match"
        vcolor  = "#ff6e6e"

    gt_html = ""
    if label is not None:
        gt_span = '<span style="color:#50fa7b">✅ Actual interaction</span>' if label == 1 else '<span style="color:#ff6e6e">❌ Negative sample</span>'
        gt_html = f"<p>Ground truth: {gt_span}</p>"

    bar_width = max(5, min(100, int(pct_score * 100)))

    return f"""
    <div style='background:#1e1e2e;border-radius:12px;padding:24px;color:#ccc;font-family:sans-serif'>
      <h3 style='color:#bd93f9;margin-top:0'>{movie_title}</h3>
      <p style='color:#888;margin:4px 0'>{genres} &nbsp;|&nbsp; {year}</p>
      <p style='color:#aaa;font-size:0.9em;margin:8px 0'>{overview}{'…' if overview else ''}</p>
      <hr style='border-color:#2a2a3a;margin:16px 0'>
      <h2 style='color:{vcolor};margin:0'>{verdict}</h2>
      <div style='display:flex;gap:32px;margin-top:16px;flex-wrap:wrap'>
        <div style='text-align:center'>
          <div style='font-size:2em;font-weight:bold;color:#f8f8f2'>{pct_score:.1%}</div>
          <div style='color:#888;font-size:0.85em'>Relevance Score</div>
        </div>
        <div style='text-align:center'>
          <div style='font-size:2em;font-weight:bold;color:#f8f8f2'>#{rank} / {n_cands}</div>
          <div style='color:#888;font-size:0.85em'>Rank among candidates</div>
        </div>
      </div>
      <div style='margin-top:16px'>
        <div style='color:#888;font-size:0.85em;margin-bottom:4px'>Relevance</div>
        <div style='background:#2a2a3a;border-radius:8px;height:12px;width:100%'>
          <div style='background:linear-gradient(90deg,#bd93f9,#ff79c6);width:{bar_width}%;height:100%;border-radius:8px'></div>
        </div>
      </div>
      {gt_html}
    </div>"""


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0d0d1a !important; font-family: 'Segoe UI', sans-serif; }
.gradio-container { max-width: 1400px !important; margin: 0 auto; }
h1, h2, h3 { color: #bd93f9 !important; }
.gr-button-primary { background: linear-gradient(135deg,#bd93f9,#ff79c6) !important; border:none !important; color:#0d0d1a !important; font-weight:bold !important; }
.gr-button-primary:hover { opacity: 0.85 !important; }
label { color: #aaa !important; }
.gr-box { background: #1e1e2e !important; border: 1px solid #2a2a3a !important; border-radius: 12px !important; }
.gr-input, .gr-dropdown { background: #1e1e2e !important; color: #f8f8f2 !important; border: 1px solid #3a3a5a !important; }
.gr-dataframe table { background: #1e1e2e !important; color: #f8f8f2 !important; }
.gr-dataframe th { background: #2a2a3a !important; color: #bd93f9 !important; }
.gr-dataframe tr:hover td { background: #2a2a3a !important; }
footer { display: none !important; }
"""


# ── UI Layout ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="🎬 Movie Recommender") as demo:

    gr.HTML("""
    <div style='text-align:center;padding:32px 0 16px;'>
      <h1 style='font-size:2.4em;background:linear-gradient(135deg,#bd93f9,#ff79c6);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin:0;font-weight:800;letter-spacing:-1px'>
        🎬 Hybrid Movie Recommender
      </h1>
      <p style='color:#888;margin:8px 0 0;font-size:1.05em'>
        SASRec · EASE · ItemKNN · BPR · LightGCN · DCN · NeuMF → LambdaRank
      </p>
      <p style='color:#555;font-size:0.9em'>ML-1M Dataset &nbsp;·&nbsp; 6,035 users &nbsp;·&nbsp; 3,807 movies</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT PANEL: user selector + history ──────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.HTML("<h3 style='color:#bd93f9;margin:0 0 8px'>👤 Select User</h3>")
            user_dd = gr.Dropdown(
                choices=all_users,
                value=all_users[0],
                label="User ID",
                filterable=True,
                container=True,
            )

            gr.HTML("<h3 style='color:#bd93f9;margin:16px 0 8px'>📜 Watch History</h3>")
            history_html = gr.HTML(value=get_history_html(all_users[0]))

        # ── MIDDLE: top-20 recommendations ───────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML("<h3 style='color:#bd93f9;margin:0 0 8px'>🏆 Top-20 Recommendations</h3>")
            metrics_html = gr.HTML(value=get_metrics_html(all_users[0]))
            recs_table = gr.Dataframe(
                value=get_top20(all_users[0]),
                headers=["Movie", "Genres", "Year", "Score"],
                interactive=False,
                wrap=True,
                height=560,
            )

        # ── RIGHT PANEL: movie selector + prediction ──────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.HTML("<h3 style='color:#bd93f9;margin:0 0 8px'>🎥 Predict for a Movie</h3>")
            init_candidates = get_candidates_for_user(all_users[0])
            movie_dd = gr.Dropdown(
                choices=init_candidates,
                value=init_candidates[0] if init_candidates else None,
                label="Select from this user's candidates (~200 movies)",
                filterable=True,
                container=True,
            )
            predict_btn = gr.Button("🔮 Get Prediction", variant="primary", size="lg")
            prediction_html = gr.HTML(
                value="<div style='color:#555;padding:20px;text-align:center'>"
                      "Select a movie and click Predict</div>"
            )

    # ── Events ───────────────────────────────────────────────────────────────
    user_dd.change(
        fn=on_user_select,
        inputs=user_dd,
        outputs=[recs_table, history_html, movie_dd, metrics_html],
    )

    predict_btn.click(
        fn=on_movie_select,
        inputs=[user_dd, movie_dd],
        outputs=prediction_html,
    )

    movie_dd.change(
        fn=on_movie_select,
        inputs=[user_dd, movie_dd],
        outputs=prediction_html,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
