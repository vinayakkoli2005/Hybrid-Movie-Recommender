"""Interactive Movie Recommender Demo — Gradio UI."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cf_pipeline.features_enhanced import (
    ENHANCED_FEAT_COLS,
    build_enhanced_feature_matrix,
    build_stats,
)

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")

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
df_all["n_cands"] = df_all.groupby("user_id")["rank"].transform("count")
df_all["pct_score"] = ((df_all["n_cands"] - df_all["rank"]) / (df_all["n_cands"] - 1)).clip(0, 1)
df_all["rank"] = df_all["rank"].round().astype(int)

items_lookup = items.set_index("item_id").to_dict(orient="index")
all_users    = sorted(train["user_id"].unique().tolist())
train_global = {
    "item_popularity": train.groupby("item_id").size().to_dict(),
    "n_train": len(train),
}
history_by_user = (
    train.sort_values("timestamp") if "timestamp" in train.columns else train
).groupby("user_id")["item_id"].apply(list).to_dict()

pipeline_metrics = {}
metrics_path = RESULTS / "tuned_pipeline.json"
if metrics_path.exists():
    pipeline_metrics = json.loads(metrics_path.read_text()).get("metrics", {})


def _compute_demo_metrics() -> dict[str, float]:
    top20 = df_all[df_all["rank"] <= 20].copy()
    if top20.empty:
        return {
            "diversity": 0.0,
            "novelty": 0.0,
            "coverage": 0.0,
            "personalization": 0.0,
        }

    # Diversity: average number of distinct genres in each user's top-20 list.
    genre_diversities = []
    for _, udf in top20.groupby("user_id"):
        genres = set()
        for iid in udf["item_id"].tolist():
            for g in items_lookup.get(int(iid), {}).get("genres", "").split("|"):
                g = g.strip()
                if g:
                    genres.add(g)
        genre_diversities.append(len(genres))
    diversity = float(np.mean(genre_diversities)) if genre_diversities else 0.0

    # Novelty: average self-information of recommended items.
    item_popularity = train_global["item_popularity"]
    n_train = train_global["n_train"]
    novelty_vals = [
        -np.log2(item_popularity.get(int(iid), 1) / n_train)
        for iid in top20["item_id"].tolist()
    ]
    novelty = float(np.mean(novelty_vals)) if novelty_vals else 0.0

    # Coverage: fraction of the catalog that appears in at least one top-20 list.
    unique_rec_items = top20["item_id"].nunique()
    total_items = max(items["item_id"].nunique(), 1)
    coverage = float(unique_rec_items / total_items)

    # Personalization: average pairwise dissimilarity across sampled user top-20 lists.
    rec_lists = (
        top20.sort_values(["user_id", "rank"])
        .groupby("user_id")["item_id"]
        .apply(lambda s: tuple(int(x) for x in s.tolist()))
        .tolist()
    )
    if len(rec_lists) > 400:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rec_lists), size=400, replace=False)
        rec_lists = [rec_lists[i] for i in idx]

    dissimilarities = []
    rec_sets = [set(lst) for lst in rec_lists]
    for i in range(len(rec_sets)):
        a = rec_sets[i]
        for j in range(i + 1, len(rec_sets)):
            b = rec_sets[j]
            union = len(a | b)
            if union == 0:
                continue
            jaccard = len(a & b) / union
            dissimilarities.append(1.0 - jaccard)
    personalization = float(np.mean(dissimilarities)) if dissimilarities else 0.0

    return {
        "diversity": diversity,
        "novelty": novelty,
        "coverage": coverage,
        "personalization": personalization,
    }


demo_metrics = _compute_demo_metrics()

print(f"Ready — {len(all_users)} users, {len(df_all)} scored pairs.")

# ── Genre pill colours (matching the React UI) ────────────────────────────────
GENRE_COLORS = {
    "Action": "#ff6e6e", "Drama": "#bd93f9", "Comedy": "#f1fa8c",
    "Crime": "#ff79c6", "Thriller": "#8be9fd", "Sci-Fi": "#50fa7b",
    "Romance": "#ff79c6", "Adventure": "#ffb86c", "Animation": "#50fa7b",
    "War": "#ff6e6e", "Mystery": "#bd93f9", "Horror": "#ff6e6e",
    "Film-Noir": "#8be9fd", "Musical": "#f1fa8c", "Fantasy": "#bd93f9",
    "Western": "#ffb86c", "Documentary": "#8be9fd", "Children's": "#f1fa8c",
}

def genre_pills(genres_str: str) -> str:
    pills = ""
    for g in genres_str.split("|"):
        g = g.strip()
        if not g:
            continue
        c = GENRE_COLORS.get(g, "#6b6b99")
        pills += (
            f"<span style='display:inline-block;padding:2px 8px;border-radius:20px;"
            f"font-size:0.65rem;font-weight:600;letter-spacing:0.04em;"
            f"background:{c}18;color:{c};border:1px solid {c}33;margin:2px 2px 0 0'>{g}</span>"
        )
    return pills

def score_bar(pct: float, color: str = "#bd93f9", height: int = 5) -> str:
    w = max(2, min(100, int(pct * 100)))
    return (
        f"<div style='background:#1c1c35;border-radius:4px;height:{height}px;width:100%;overflow:hidden;margin-top:6px'>"
        f"<div style='height:100%;border-radius:4px;width:{w}%;"
        f"background:linear-gradient(90deg,{color},#ff79c6);"
        f"box-shadow:0 0 8px {color}55'></div></div>"
    )


# ── Data functions ────────────────────────────────────────────────────────────
def get_candidates_for_user(user_id: int) -> list[str]:
    udf = df_all[df_all["user_id"] == user_id].sort_values("rank")
    titles = []
    for iid in udf["item_id"].tolist():
        meta = items_lookup.get(int(iid), {})
        titles.append(meta.get("title", f"Item {iid}"))
    return titles


def get_top_summary_html(user_id: int) -> str:
    udf = df_all[df_all["user_id"] == user_id].copy()
    if udf.empty:
        return ""
    top = udf.nsmallest(20, "rank")

    all_genres = set()
    for iid in top["item_id"].tolist():
        for g in items_lookup.get(int(iid), {}).get("genres", "").split("|"):
            if g.strip():
                all_genres.add(g.strip())
    diversity = len(all_genres)

    item_popularity = train_global["item_popularity"]
    n_train = train_global["n_train"]
    novelties = [
        -np.log2(item_popularity.get(int(iid), 1) / n_train)
        for iid in top["item_id"].tolist()
    ]
    avg_novelty = float(np.mean(novelties)) if novelties else 0.0

    def card(icon, label, value, color):
        return (
            f"<div style='flex:1;min-width:120px;background:#14142a;border-radius:12px;"
            f"padding:14px 16px;border:1px solid {color}22;box-shadow:0 4px 20px {color}11'>"
            f"<div style='font-size:1.1rem;margin-bottom:4px'>{icon}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{color};"
            f"font-family:\"Space Mono\",monospace;letter-spacing:-0.5px'>{value}</div>"
            f"<div style='font-size:0.68rem;color:#6b6b99;margin-top:3px;"
            f"text-transform:uppercase;letter-spacing:0.05em'>{label}</div>"
            f"</div>"
        )

    cards = (
        card("◆", "Genre Diversity", str(diversity), "#ff79c6") +
        card("◇", "Avg Novelty", f"{avg_novelty:.2f}", "#8be9fd")
    )
    return f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px'>{cards}</div>"


def get_demo_metrics_cards_html() -> str:
    cards = [
        {
            "icon": "◆",
            "label": "Diversity",
            "value": f"{demo_metrics['diversity']:.1f}",
            "color": "#ff79c6",
            "desc": "Average number of different genres appearing in each user's top-20 list.",
        },
        {
            "icon": "◇",
            "label": "Novelty",
            "value": f"{demo_metrics['novelty']:.2f}",
            "color": "#8be9fd",
            "desc": "Higher means recommendations are less mainstream and more discovery-oriented.",
        },
        {
            "icon": "▣",
            "label": "Coverage",
            "value": f"{demo_metrics['coverage']:.0%}",
            "color": "#bd93f9",
            "desc": "Share of the movie catalog that shows up in at least one user's recommendations.",
        },
        {
            "icon": "◎",
            "label": "Personalization",
            "value": f"{demo_metrics['personalization']:.2f}",
            "color": "#50fa7b",
            "desc": "Higher means different users receive more different recommendation lists.",
        },
    ]

    html_cards = []
    for card in cards:
        html_cards.append(
            f"<div style='flex:1;min-width:220px;background:#14142a;border-radius:14px;"
            f"padding:16px 18px;border:1px solid {card['color']}22;box-shadow:0 4px 20px {card['color']}11'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
            f"<span style='font-size:1rem'>{card['icon']}</span>"
            f"<span style='font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#e8e8ff'>{card['label']}</span>"
            f"</div>"
            f"<div style='font-size:1.55rem;font-weight:700;color:{card['color']};font-family:\"Space Mono\",monospace;margin-bottom:8px'>{card['value']}</div>"
            f"<div style='font-size:0.72rem;line-height:1.5;color:#6b6b99'>{card['desc']}</div>"
            f"</div>"
        )
    return (
        "<div style='margin-top:14px;margin-bottom:14px'>"
        "<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#e8e8ff;margin-bottom:10px'>Demo Metrics</div>"
        f"<div style='display:flex;gap:10px;flex-wrap:wrap'>{''.join(html_cards)}</div>"
        "</div>"
    )


def get_overall_metrics_graph():
    ks = [1, 5, 10, 20]
    hr_vals = [float(pipeline_metrics.get(f"HR@{k}", 0.0)) for k in ks]
    ndcg_vals = [float(pipeline_metrics.get(f"NDCG@{k}", 0.0)) for k in ks]
    x_labels = [f"@{k}" for k in ks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=hr_vals,
        mode="lines+markers+text",
        name="HR",
        text=[f"{v:.3f}" for v in hr_vals],
        textposition="top center",
        line=dict(color="#bd93f9", width=4, shape="spline"),
        marker=dict(size=12, color="#bd93f9", line=dict(color="#ffffff", width=1)),
        hovertemplate="HR %{x}: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=ndcg_vals,
        mode="lines+markers+text",
        name="NDCG",
        text=[f"{v:.3f}" for v in ndcg_vals],
        textposition="bottom center",
        line=dict(color="#50fa7b", width=4, shape="spline"),
        marker=dict(size=12, color="#50fa7b", line=dict(color="#ffffff", width=1)),
        hovertemplate="NDCG %{x}: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Average HR / NDCG",
            x=0.02,
            font=dict(size=18, color="#e8e8ff"),
        ),
        paper_bgcolor="#0e0e1f",
        plot_bgcolor="#14142a",
        font=dict(family="Space Grotesk, sans-serif", color="#e8e8ff"),
        height=360,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title="Cutoff K",
        showgrid=False,
        linecolor="rgba(189,147,249,0.2)",
        tickfont=dict(color="#e8e8ff"),
    )
    fig.update_yaxes(
        title="Score",
        range=[0, 1.0],
        gridcolor="rgba(189,147,249,0.12)",
        zeroline=False,
        tickfont=dict(color="#e8e8ff"),
    )
    return fig


def get_history_html(user_id: int) -> str:
    hist = history_by_user.get(user_id, [])[-15:]
    if not hist:
        return "<p style='color:#6b6b99;font-size:0.85rem'>No history found.</p>"
    n_ratings = len(history_by_user.get(user_id, []))
    header = (
        f"<div style='background:#14142a;border-radius:12px;padding:10px 12px;"
        f"border:1px solid rgba(189,147,249,0.12);margin-bottom:10px;display:flex;align-items:center;gap:10px'>"
        f"<div style='width:36px;height:36px;border-radius:50%;"
        f"background:linear-gradient(135deg,#bd93f9,#ff79c6);"
        f"display:flex;align-items:center;justify-content:center;"
        f"font-size:0.7rem;font-weight:700;font-family:\"Space Mono\",monospace;color:#07071a'>{user_id}</div>"
        f"<div><div style='font-size:0.82rem;font-weight:600;color:#e8e8ff'>User #{user_id}</div>"
        f"<div style='font-size:0.68rem;color:#6b6b99'>{n_ratings} ratings</div></div></div>"
    )
    lines = [header]
    for iid in reversed(hist):
        meta   = items_lookup.get(iid, {})
        title  = meta.get("title", f"Item {iid}")
        genres = meta.get("genres", "")
        year   = title[-5:-1] if len(title) > 6 and title[-1] == ")" else "—"
        short  = title.replace(f" ({year})", "") if year != "—" else title
        lines.append(
            f"<div style='padding:10px 12px;border-radius:10px;background:#14142a;"
            f"border:1px solid rgba(189,147,249,0.12);margin-bottom:6px'>"
            f"<div style='display:flex;align-items:flex-start;gap:8px'>"
            f"<div style='width:32px;height:32px;border-radius:6px;flex-shrink:0;"
            f"background:#1c1c35;display:flex;align-items:center;justify-content:center;"
            f"font-size:0.62rem;color:#6b6b99;font-family:\"Space Mono\",monospace'>{year}</div>"
            f"<div style='flex:1;min-width:0'>"
            f"<div style='font-size:0.78rem;font-weight:600;color:#e8e8ff;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{short}</div>"
            f"<div style='margin-top:4px'>{genre_pills(genres)}</div>"
            f"</div></div></div>"
        )
    return "".join(lines)


def get_recs_html(user_id: int) -> str:
    udf = df_all[df_all["user_id"] == user_id].copy()
    if udf.empty:
        return "<p style='color:#6b6b99'>No recommendations found.</p>"
    top = udf.nsmallest(20, "rank")
    cards = []
    for i, (_, row) in enumerate(top.iterrows()):
        meta      = items_lookup.get(int(row["item_id"]), {})
        title     = meta.get("title", f"Item {int(row['item_id'])}")
        genres    = meta.get("genres", "")
        year      = title[-5:-1] if len(title) > 6 and title[-1] == ")" else "—"
        short     = title.replace(f" ({year})", "") if year != "—" else title
        rank      = int(row["rank"])
        pct       = float(row["pct_score"])
        dot_color = "#50fa7b" if rank <= 5 else "#f1fa8c" if rank <= 20 else "#ff6e6e"
        num_bg    = "linear-gradient(135deg,#bd93f9,#ff79c6)" if i == 0 else "#222240"
        num_color = "#07071a" if i == 0 else "#6b6b99"

        cards.append(
            f"<div style='padding:12px 14px;border-radius:12px;background:#14142a;"
            f"border:1px solid rgba(189,147,249,0.12);margin-bottom:6px'>"
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<div style='flex:1;min-width:0'>"
            f"<div style='display:flex;align-items:center;justify-content:space-between;gap:8px'>"
            f"<div style='font-size:0.82rem;font-weight:600;color:#e8e8ff;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1'>"
            f"{short}<span style='color:#6b6b99;font-weight:400;font-size:0.7rem;margin-left:4px'>({year})</span></div>"
            f"<div style='flex-shrink:0;display:flex;align-items:center;gap:5px'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{dot_color};"
            f"box-shadow:0 0 6px {dot_color}'></div>"
            f"<span style='font-size:0.72rem;font-family:\"Space Mono\",monospace;color:#6b6b99'>{pct:.0%}</span>"
            f"</div></div>"
            f"{score_bar(pct)}"
            f"<div style='margin-top:6px'>{genre_pills(genres)}</div>"
            f"</div></div></div>"
        )
    return "".join(cards)


def on_user_select(user_id: int):
    summary_html = get_top_summary_html(user_id)
    history_html = get_history_html(user_id)
    recs_html    = get_recs_html(user_id)
    candidates   = get_candidates_for_user(user_id)
    pred_placeholder = (
        "<div style='display:flex;align-items:center;justify-content:center;"
        "flex-direction:column;gap:12px;color:#6b6b99;padding:40px 20px'>"
        "<div style='font-size:2.5rem;opacity:0.3'>🎬</div>"
        "<div style='font-size:0.82rem'>Select a movie to see prediction details</div></div>"
    )
    return (
        summary_html,
        recs_html,
        history_html,
        gr.update(choices=candidates, value=candidates[0] if candidates else None),
        pred_placeholder,
    )


def on_movie_select(user_id: int, movie_title: str) -> str:
    if not movie_title:
        return ""
    item_id = None
    for iid, meta in items_lookup.items():
        if meta.get("title") == movie_title:
            item_id = iid
            break
    if item_id is None:
        return "<p style='color:#ff6e6e'>Movie not found.</p>"

    row      = df_all[(df_all["user_id"] == user_id) & (df_all["item_id"] == item_id)]
    meta     = items_lookup.get(item_id, {})
    genres   = meta.get("genres", "?")
    overview = meta.get("overview", "")[:300]
    year     = meta.get("release_date", "?")

    if row.empty:
        return (
            f"<div style='background:#14142a;border-radius:16px;padding:20px;"
            f"border:1px solid rgba(255,110,110,0.2)'>"
            f"<div style='color:#ff6e6e;font-weight:700;margin-bottom:8px'>⚠ Not in candidate set</div>"
            f"<div style='color:#e8e8ff;font-size:0.88rem;font-weight:600'>{movie_title}</div>"
            f"<div style='color:#6b6b99;font-size:0.78rem;margin-top:4px'>{genres.replace('|',' · ')}</div>"
            f"<div style='color:#6b6b99;font-size:0.78rem;margin-top:8px;line-height:1.5'>"
            f"This movie was not among the top candidates generated by the CF models for this user.</div>"
            f"</div>"
        )

    rank      = int(row["rank"].values[0])
    n_cands   = int(row["n_cands"].values[0])
    pct_score = float(row["pct_score"].values[0])
    label     = int(row["label"].values[0]) if "label" in row.columns else None

    if rank <= 5:
        verdict, vcolor, vicon = "Highly Recommended", "#50fa7b", "🟢"
    elif rank <= 20:
        verdict, vcolor, vicon = "Likely to Enjoy",    "#f1fa8c", "🟡"
    else:
        verdict, vcolor, vicon = "Unlikely Match",     "#ff6e6e", "🔴"

    gt_html = ""
    if label is not None:
        gt_span = (
            '<span style="color:#50fa7b">✅ Actual interaction</span>'
            if label == 1 else
            '<span style="color:#ff6e6e">❌ Negative sample</span>'
        )
        gt_html = (
            f"<div style='margin-top:12px;padding:8px 12px;border-radius:8px;"
            f"background:rgba(80,250,123,0.08);border:1px solid rgba(80,250,123,0.2);"
            f"font-size:0.75rem;color:#e8e8ff'>Ground truth: {gt_span}</div>"
        )

    short = movie_title.replace(f" ({year[:4]})", "") if year and len(year) >= 4 else movie_title
    bar_w = max(2, min(100, int(pct_score * 100)))
    rank_w = max(2, min(100, int((1 - (rank - 1) / n_cands) * 100)))

    return (
        f"<div style='background:#14142a;border-radius:16px;padding:20px;"
        f"border:1px solid rgba(189,147,249,0.15);"
        f"background:linear-gradient(135deg,#14142a 0%,rgba(189,147,249,0.04) 100%)'>"
        # title
        f"<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:10px'>"
        f"<div><div style='font-size:1.05rem;font-weight:700;color:#e8e8ff;line-height:1.3'>{short}</div>"
        f"<div style='font-size:0.75rem;color:#6b6b99;margin-top:2px'>{year}</div></div>"
        f"<div style='background:#1c1c35;border-radius:8px;padding:5px 10px;"
        f"font-size:0.68rem;font-family:\"Space Mono\",monospace;color:#bd93f9;"
        f"font-weight:600;flex-shrink:0;border:1px solid rgba(189,147,249,0.2)'>#{rank}/{n_cands}</div>"
        f"</div>"
        # genres
        f"<div style='margin-bottom:10px'>{genre_pills(genres)}</div>"
        # overview
        + (f"<div style='font-size:0.78rem;color:#6b6b99;line-height:1.6;margin-bottom:14px;"
           f"display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden'>"
           f"{overview}{'…' if overview else ''}</div>" if overview else "") +
        # verdict
        f"<div style='background:#0e0e1f;border-radius:10px;padding:10px 14px;"
        f"border:1px solid {vcolor}33;display:inline-flex;align-items:center;gap:8px;margin-bottom:14px'>"
        f"<span>{vicon}</span><span style='color:{vcolor};font-weight:700;font-size:0.95rem'>{verdict}</span></div>"
        # score cards
        f"<div style='display:flex;gap:10px;margin-bottom:14px'>"
        f"<div style='flex:1;background:#0e0e1f;border-radius:10px;padding:12px;text-align:center;border:1px solid rgba(189,147,249,0.12)'>"
        f"<div style='font-size:1.4rem;font-weight:700;color:{vcolor};font-family:\"Space Mono\",monospace'>{pct_score:.1%}</div>"
        f"<div style='font-size:0.68rem;color:#6b6b99;margin-top:2px;text-transform:uppercase;letter-spacing:0.05em'>Relevance</div></div>"
        f"<div style='flex:1;background:#0e0e1f;border-radius:10px;padding:12px;text-align:center;border:1px solid rgba(189,147,249,0.12)'>"
        f"<div style='font-size:1.4rem;font-weight:700;color:#e8e8ff;font-family:\"Space Mono\",monospace'>#{rank}</div>"
        f"<div style='font-size:0.68rem;color:#6b6b99;margin-top:2px;text-transform:uppercase;letter-spacing:0.05em'>Rank</div></div>"
        f"</div>"
        # bars
        f"<div style='margin-bottom:4px'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
        f"<span style='font-size:0.68rem;color:#6b6b99;text-transform:uppercase;letter-spacing:0.05em'>Relevance Score</span>"
        f"<span style='font-size:0.68rem;font-family:\"Space Mono\",monospace;color:{vcolor}'>{pct_score:.1%}</span></div>"
        f"<div style='background:#1c1c35;border-radius:4px;height:6px;overflow:hidden'>"
        f"<div style='height:100%;width:{bar_w}%;background:linear-gradient(90deg,{vcolor},#ff79c6);border-radius:4px'></div></div></div>"
        f"<div style='margin-top:10px'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
        f"<span style='font-size:0.68rem;color:#6b6b99;text-transform:uppercase;letter-spacing:0.05em'>Rank Position</span>"
        f"<span style='font-size:0.68rem;font-family:\"Space Mono\",monospace;color:#8be9fd'>#{rank} of {n_cands}</span></div>"
        f"<div style='background:#1c1c35;border-radius:4px;height:6px;overflow:hidden'>"
        f"<div style='height:100%;width:{rank_w}%;background:linear-gradient(90deg,#8be9fd,#bd93f9);border-radius:4px'></div></div></div>"
        + gt_html +
        f"</div>"
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .gradio-container {
  background: #070712 !important;
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif !important;
  color: #e8e8ff !important;
}
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; }

/* scrollbars */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3d3d60; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #bd93f9; }

/* dropdowns & inputs */
input, select, textarea, .gr-input, .gr-dropdown, [data-testid="dropdown"] {
  background: #14142a !important;
  border: 1px solid rgba(189,147,249,0.18) !important;
  border-radius: 10px !important;
  color: #e8e8ff !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.88rem !important;
}
input:focus, select:focus {
  border-color: #bd93f9 !important;
  box-shadow: 0 0 0 3px rgba(189,147,249,0.15) !important;
  outline: none !important;
}

/* labels */
label, .gr-label {
  color: #6b6b99 !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
}

/* primary button */
button.primary, .gr-button-primary, button[variant="primary"] {
  background: linear-gradient(135deg, #bd93f9, #ff79c6) !important;
  border: none !important;
  color: #07071a !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: 0.02em !important;
  transition: opacity 0.2s !important;
}
button.primary:hover, .gr-button-primary:hover { opacity: 0.85 !important; }

/* panels */
.gr-box, .gr-form, .gr-panel {
  background: #0e0e1f !important;
  border: 1px solid rgba(189,147,249,0.12) !important;
  border-radius: 16px !important;
}

/* section headers */
h1, h2, h3 { color: #bd93f9 !important; }

footer { display: none !important; }

/* outer page bg */
.main { background: #070712 !important; }
.contain { background: #070712 !important; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Hybrid Movie Recommender") as demo:

    # Header
    gr.HTML("""
    <div style='padding:14px 24px 12px;border-bottom:1px solid rgba(189,147,249,0.12);
                background:rgba(7,7,18,0.97);display:flex;align-items:center;
                justify-content:space-between;flex-wrap:wrap;gap:8px'>
      <div style='display:flex;align-items:center;gap:12px'>
        <div style='background:linear-gradient(135deg,#bd93f9,#ff79c6);border-radius:8px;
                    width:30px;height:30px;display:flex;align-items:center;
                    justify-content:center;font-size:0.95rem'>🎬</div>
        <div>
          <div style='font-size:0.95rem;font-weight:700;letter-spacing:-0.2px;color:#e8e8ff'>
            Hybrid Movie Recommender</div>
          <div style='font-size:0.6rem;color:#6b6b99;letter-spacing:0.06em;text-transform:uppercase'>
            SASRec · EASE · ItemKNN · BPR · LightGCN · DCN · NeuMF → LambdaRank</div>
        </div>
      </div>
      <div style='font-size:0.68rem;color:#6b6b99;font-family:"Space Mono",monospace'>
        ML-1M &nbsp;·&nbsp; 6,035 users &nbsp;·&nbsp; 3,807 movies</div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: user + history ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=240):
            gr.HTML(
                "<div style='display:flex;align-items:center;gap:8px;margin:12px 0 8px'>"
                "<div style='width:6px;height:6px;border-radius:50%;background:#bd93f9;"
                "box-shadow:0 0 8px #bd93f9'></div>"
                "<span style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                "letter-spacing:0.08em;color:#6b6b99'>Select User</span></div>"
            )
            user_dd = gr.Dropdown(
                choices=all_users,
                value=all_users[0],
                label="User ID",
                filterable=True,
                container=True,
            )
            gr.HTML(
                "<div style='display:flex;align-items:center;gap:8px;margin:16px 0 8px'>"
                "<div style='width:6px;height:6px;border-radius:50%;background:#bd93f9;"
                "box-shadow:0 0 8px #bd93f9'></div>"
                "<span style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                "letter-spacing:0.08em;color:#6b6b99'>Watch History</span></div>"
            )
            history_html = gr.HTML(value=get_history_html(all_users[0]))

        # ── MIDDLE: metrics + recs ────────────────────────────────────────────
        with gr.Column(scale=2):
            summary_html = gr.HTML(value=get_top_summary_html(all_users[0]))
            gr.HTML(
                "<div style='display:flex;align-items:center;gap:8px;margin:10px 0 8px'>"
                "<div style='width:6px;height:6px;border-radius:50%;background:#bd93f9;"
                "box-shadow:0 0 8px #bd93f9'></div>"
                "<span style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                "letter-spacing:0.08em;color:#6b6b99'>Top-20 Recommendations</span></div>"
            )
            recs_html = gr.HTML(value=get_recs_html(all_users[0]))

        # ── RIGHT: movie picker + prediction ─────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.HTML(
                "<div style='display:flex;align-items:center;gap:8px;margin:12px 0 8px'>"
                "<div style='width:6px;height:6px;border-radius:50%;background:#ff79c6;"
                "box-shadow:0 0 8px #ff79c6'></div>"
                "<span style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                "letter-spacing:0.08em;color:#6b6b99'>Prediction Details</span></div>"
            )
            init_candidates = get_candidates_for_user(all_users[0])
            movie_dd = gr.Dropdown(
                choices=init_candidates,
                value=init_candidates[0] if init_candidates else None,
                label="Select from this user's candidates",
                filterable=True,
                container=True,
            )
            predict_btn = gr.Button("🔮 Get Prediction", variant="primary", size="lg")
            prediction_html = gr.HTML(
                value=(
                    "<div style='display:flex;align-items:center;justify-content:center;"
                    "flex-direction:column;gap:12px;color:#6b6b99;padding:40px 20px'>"
                    "<div style='font-size:2.5rem;opacity:0.3'>🎬</div>"
                    "<div style='font-size:0.82rem'>Select a movie to see prediction details</div></div>"
                )
            )

    with gr.Row():
        demo_metrics_html = gr.HTML(value=get_demo_metrics_cards_html())

    with gr.Row():
        overall_metrics_plot = gr.Plot(value=get_overall_metrics_graph())

    # ── Events ───────────────────────────────────────────────────────────────
    user_dd.change(
        fn=on_user_select,
        inputs=user_dd,
        outputs=[summary_html, recs_html, history_html, movie_dd, prediction_html],
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
        show_error=True,
        css=CSS,
    )
    while True:
        time.sleep(3600)
