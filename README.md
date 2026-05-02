---
title: Hybrid Movie Recommender
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---

# Hybrid Movie Recommendation System

End-to-end multi-stage recommendation pipeline on MovieLens-1M.

**Pipeline:** 7-model ensemble (BPR, LightGCN, NeuMF, SASRec, EASE, DCN, Popularity) → 13-dim feature vector → LightGBM LambdaRank meta-learner tuned with Optuna.

**LLM stage:** Gemma-2 fine-tuned with LoRA; Yes-token logit probabilities used as semantic features.

**Metrics:** HR@10 = 0.797 · NDCG@10 = 0.636 · Coverage = 78% · Personalisation = 0.99
