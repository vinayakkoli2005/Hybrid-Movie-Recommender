# Hybrid Movie Recommendation System

<p align="center">
  <img src="https://i.ibb.co/sJNgbqdP/ui-main.jpg" alt="Hybrid Recommender UI" width="100%"/>
</p>

<p align="center">
  <b>A production-grade, multi-stage recommendation pipeline</b><br/>
  7 collaborative filtering models · LoRA-fine-tuned LLM reranking · LightGBM LambdaRank meta-learner · Gradio interactive demo
</p>

<p align="center">
  <img src="https://img.shields.io/badge/HR%4010-0.797-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/NDCG%4010-0.636-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Personalisation-0.99-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Catalog%20Coverage-78%25-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-%3E%3D3.11-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square"/>
</p>

---

## Overview

This repository implements a **complete, end-to-end hybrid recommendation pipeline** trained and evaluated on [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) (1M explicit ratings, 6,040 users, 3,952 movies), enriched with a [Kaggle 45K movie metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) source for richer content-level features.

The system is architected as a **three-stage cascade**:

1. **Candidate Generation** — Seven independent CF and sequential models produce diverse top-K candidate lists per user.
2. **LLM Semantic Augmentation** — A LoRA-adapted Gemma model scores each (user, candidate) pair using enriched natural-language metadata, bridging the semantic gap that co-occurrence signals alone cannot capture.
3. **Meta-Learning (Reranking)** — A LightGBM LambdaRank model trained with Optuna HPO combines all signals into a 13-dimensional feature vector and directly optimises NDCG to produce final ranked lists.

The result is a system that simultaneously achieves **strong ranking accuracy** (HR@10 = 0.797), **near-perfect personalisation** (0.99 inter-user list diversity), and **78% catalog coverage** — a combination that popularity-dominated or single-model approaches consistently fail to deliver.

---

## Architecture

<p align="center">
  <img src="https://i.ibb.co/m5dNSkZw/pipeline-flowchart.png" alt="End-to-end Pipeline Architecture" width="85%"/>
</p>

<p align="center"><i>Figure 1: End-to-end pipeline. Data flows top-to-bottom through five major stages — Preprocessing, Candidate Generation, LLM Feature Extraction (optional), Feature Engineering, Meta-Learner HPO + Training, and the Gradio UI.</i></p>

### Stage 1 — Data Preparation

Raw MovieLens ratings are converted to **implicit binary feedback** (positive if rating ≥ 4, otherwise discarded). No mean-filling or imputation is applied; the system treats all downstream data as a set of `(user, item, timestamp)` triples.

A **leave-one-out** temporal split is used:
- Training: all but the last 2 interactions per user
- Validation: second-to-last interaction
- Test: last interaction

For evaluation, each user is ranked against a frozen pool of **1 positive + 99 randomly sampled negatives**, stored deterministically for fully reproducible metric computation.

### Stage 2 — Candidate Generation (7 Models)

All models run independently and produce columnar `cf_scores_<model>.parquet` files. No model has visibility into another's scores at this stage.

| Model | Type | Key Idea |
|---|---|---|
| **Popularity** | Non-personalised baseline | Global interaction frequency ranking |
| **ItemKNN** | Neighbourhood CF | Cosine similarity over binary user-item vectors |
| **EASE^R** | Shallow autoencoder | Closed-form item-to-item weight matrix; zero latent factors |
| **BPR** | Matrix factorisation | Pairwise ranking loss maximising P(positive > negative) |
| **LightGCN** | Graph neural network | Multi-layer graph convolution on bipartite user-item graph, no feature transformation |
| **DCN v2** | Deep & Cross Network | Explicit bounded-degree feature interactions + deep layers |
| **NeuMF** | Neural MF | GMF branch + MLP branch concatenated before output |
| **SASRec** | Sequential transformer | Causal self-attention over user interaction history |

### Stage 3 — LLM Semantic Feature (Optional)

A **Gemma** base model is fine-tuned using **LoRA** (rank decomposition, only A/B matrices trained, base weights frozen) on a synthetic dataset of `(user_history, candidate_item, yes/no)` pairs constructed from MovieLens + Kaggle 45K metadata.

During inference, the logit ratio `softmax(l_yes, l_no)` yields a scalar `p_yes(u,i) ∈ [0,1]` — a semantic compatibility score that is **orthogonal** to all co-occurrence-based CF signals.

### Stage 4 — Feature Engineering & Rank Normalisation

Raw CF scores are on incompatible scales (BPR dot products vs. EASE predictions). Instead of min-max normalisation (sensitive to outliers), we use **rank normalisation**:

```
s̃_m(u, i) = (|C_u| - rank_m(u, i) + 1) / |C_u|  ∈ (0, 1]
```

This maps each model's output to a uniform percentile scale, ensuring fair signal combination regardless of score magnitude. The complete **13-dimensional feature vector** per (user, candidate) pair:

| # | Feature | Source |
|---|---|---|
| 1–8 | `{bpr,itemknn,ease,pop,lightgcn,dcn,neumf,sasrec}_rank_norm` | Rank-normalised CF scores |
| 9 | `user_interaction_count` | User training history length |
| 10 | `user_avg_item_popularity` | Mean global popularity of user's watched items |
| 11 | `item_interaction_count` | Item's global training interaction count |
| 12 | `item_popularity_rank` | Item's rank in the global popularity ordering |
| 13 | `llm_yes_prob` | LoRA-Gemma semantic score |

### Stage 5 — LightGBM LambdaRank Meta-Learner

Standard regression/classification objectives do not optimise ranking metrics directly. **LambdaRank** computes pseudo-gradients that approximate the change in NDCG from swapping item pairs:

```
λ_ij ≈ |ΔNDCG_ij| / (1 + exp(ŷ_ui - ŷ_uj))
```

LightGBM's GBDT implementation of these gradients is extremely fast on the 13-dimensional tabular feature matrix, and naturally handles the grouped-by-user structure of the ranking problem.

**Hyperparameter optimisation** uses Optuna with a Tree-structured Parzen Estimator (TPE) sampler over 8 parameters (learning rate, num_leaves, max_depth, min_child_samples, λ_L1, λ_L2, feature_fraction, bagging_fraction). The study is persisted in SQLite and can be paused/resumed without re-running completed trials.

---

## Results

### Ranking Accuracy

| Metric | @1 | @5 | @10 | @20 |
|---|---|---|---|---|
| **HR** | 0.530 | 0.618 | **0.797** | 0.915 |
| **NDCG** | 0.530 | 0.574 | **0.636** | 0.665 |

> HR@1 = NDCG@1 since both reduce to a single top-position binary check.

### Beyond Accuracy

| Metric | Value | Interpretation |
|---|---|---|
| Genre Diversity (avg @20) | **11.6 unique genres/list** | High intra-list diversity; ensemble models surface varied content |
| Novelty (avg @20) | **10.93** | Balances mainstream and long-tail; rank normalisation prevents popularity dominance |
| Catalog Coverage (@20) | **78%** | 2,970 of 3,807 catalog items appear in at least one user's top-20 |
| Personalisation (@20) | **0.99** | Near-zero inter-user list overlap across 6,000+ users — no two users get the same recommendations |

<p align="center">
  <img src="https://i.ibb.co/GgyXQ1M/ui-metrics.jpg" alt="Metrics Dashboard" width="90%"/>
</p>

<p align="center"><i>Figure 2: Live metrics dashboard in the Gradio UI. Diversity, novelty, coverage, and personalisation cards alongside the HR/NDCG@K curve.</i></p>

---

## Why This Pipeline Works

**1. Ensemble diversity is the primary driver of coverage.**  
BPR and ItemKNN tend to recommend genre-similar items. SASRec and LightGCN surface structurally different candidates based on sequential patterns and graph topology. The meta-learner learns to combine these complementary signals rather than deferring to any single model.

**2. Rank normalisation is non-negotiable.**  
A BPR dot-product of 1.2 and an EASE prediction of 0.04 are not comparable. Normalising by rank instead of raw value prevents any single model from dominating the feature space due to scale.

**3. LLM signals are orthogonal, not redundant.**  
CF models exploit co-occurrence patterns — they know *who* tends to like *what*, but not *why*. The LoRA-Gemma semantic score operates on item content (plot, genre, metadata) and user history in natural language. On the 13-dimensional feature matrix, `llm_yes_prob` is the feature with the lowest collinearity with all CF signals.

**4. LambdaRank directly optimises what you measure.**  
Training a binary classifier or regressor on relevance labels and then evaluating with NDCG creates a train/eval gap. LambdaRank eliminates this by computing gradients in the direction that maximally improves NDCG.

---

## Quickstart

### Requirements

```
Python >= 3.11
torch, lightgbm, transformers, peft, accelerate
optuna, hydra-core, omegaconf
pandas, pyarrow, numpy
gradio
pytest
```

Install with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/vinayakkoli2005/CF-PROJECT.git
cd CF-PROJECT
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### Data

Place the raw MovieLens-1M files under `data/raw/ml-1m/`:

```
data/raw/ml-1m/
├── ratings.dat
├── movies.dat
└── users.dat
```

The Kaggle 45K enrichment dataset (`movies_metadata.csv`) goes under `data/raw/kaggle/`.

---

## Reproducing Results

All steps are orchestrated through individual scripts. Each stage writes canonical artifacts to `data/processed/`, so any stage can be rerun independently.

```bash
# 1. Binarise ratings, leave-one-out split, freeze 99 negatives per user
uv run python scripts/prepare_data.py

# 2. Train all CF models and dump candidate scores
CUDA_VISIBLE_DEVICES=0 uv run python scripts/dump_scores.py

# 3. (Optional) Build LoRA fine-tuning dataset from user histories + metadata
uv run python scripts/build_lora_dataset.py

# 4. (Optional) Fine-tune Gemma adapter
uv run python scripts/lora_train.py

# 5. (Optional) Generate LLM yes-probability features
uv run python scripts/build_llm_features_lora.py

# 6. Hyperparameter optimisation for LightGBM LambdaRank (resumable)
uv run python scripts/tune_meta_learner.py

# 7. Train final model on train + val with best Optuna parameters
uv run python scripts/train_final_model.py

# 8. Evaluate and write results/{hybrid_pipeline,ablation}.json
uv run python scripts/run_pipeline.py

# 9. Launch the interactive Gradio UI on port 7860
uv run python app.py
```

For a single-command rebuild from scratch:

```bash
bash scripts/rebuild_all.sh
```

For fast recovery (skipping already-complete stages based on artifact presence):

```bash
bash scripts/fast_recover.sh
```

### Run a Single Experiment

```bash
uv run python scripts/eval.py +experiment=baseline_pop
uv run python scripts/eval.py +experiment=lightgcn
uv run python scripts/eval.py +experiment=hybrid_full
```

---

## Repository Layout

```
CF-PROJECT/
├── app.py                          # Gradio UI entry point
├── configs/
│   ├── config.yaml                 # Hydra master config
│   ├── data/ml1m.yaml              # Binarisation threshold, split params
│   ├── eval/ncf_protocol.yaml      # Leave-one-out, K cutoffs, 99 negatives
│   └── experiment/*.yaml           # Per-model hyperparameters
├── scripts/
│   ├── prepare_data.py             # Stage 1 — preprocessing
│   ├── dump_scores.py              # Stage 2 — all CF model scores
│   ├── build_lora_dataset.py       # Stage 3a — LoRA training data
│   ├── lora_train.py               # Stage 3b — Gemma LoRA adapter
│   ├── build_llm_features_lora.py  # Stage 3c — LLM inference → parquet
│   ├── tune_meta_learner.py        # Stage 4a — Optuna HPO
│   ├── train_final_model.py        # Stage 4b — final LightGBM model
│   ├── run_pipeline.py             # Stage 5 — evaluation
│   ├── rebuild_all.sh              # Full pipeline orchestrator
│   └── fast_recover.sh             # Skip completed stages
├── src/cf_pipeline/
│   ├── data/
│   │   ├── loaders.py              # MovieLens + Kaggle data loaders
│   │   └── binarize.py             # binarize_ratings(df, threshold=4)
│   └── models/                     # ItemKNN, EASE, BPR, LightGCN, DCN, NeuMF, SASRec
├── data/
│   ├── raw/                        # Input data (not tracked by git)
│   └── processed/                  # Pipeline artifacts (parquet, json)
├── checkpoints/
│   ├── meta_lgbm.pkl               # Initial LightGBM model
│   ├── meta_lgbm_tuned.pkl         # Optuna-tuned final model
│   ├── optuna_study.db             # Resumable SQLite study
│   └── lora/                       # LoRA adapter checkpoints
├── results/
│   ├── hybrid_pipeline.json        # Final HR/NDCG results
│   ├── ablation.json               # Ablation study results
│   ├── best_params.json            # Best Optuna parameters
│   └── table1_baselines.md         # Baseline comparison table
├── tests/
│   ├── data/                       # Loader + binarisation unit tests
│   ├── metrics/                    # HR, NDCG, MAP, MAR tests
│   ├── models/                     # NeuMF, SASRec, LightGCN component tests
│   └── integration/                # End-to-end smoke tests
└── docs/superpowers/
    ├── specs/                      # Design documents
    └── plans/                      # Build plan
```

---

## Processed Artifacts

| File | Description |
|---|---|
| `train.parquet` | Implicit positive interactions for training |
| `val.parquet` | Second-to-last interaction per user (validation) |
| `test.parquet` | Last interaction per user (held-out test) |
| `items_metadata.parquet` | Title, genres, TMDB links, Kaggle enrichment |
| `eval_negatives.json` | Frozen 99 negatives + 1 positive per user per split |
| `cf_scores_*.parquet` | Per-model candidate scores (one file per model) |
| `llm_features.parquet` | LLM `yes_prob` per (user, candidate) pair |
| `lora_train.jsonl` | LoRA fine-tuning prompt/response pairs |

---

## Gradio UI

The interactive demo (`app.py`) precomputes and caches ranked lists for all users at startup. It exposes three panels:

**Left** — User selector dropdown + scrollable watch history with year and genre tags.  
**Centre** — Top-20 recommendations with relevance score bars, genre tags, and live diversity/novelty cards.  
**Right** — Per-(user, candidate) prediction details: relevance score, rank within candidate pool, feature breakdown, ground-truth label.

A HR/NDCG@K line chart updates dynamically and shows the accuracy/position tradeoff across cutoffs K ∈ {1, 5, 10, 20}.

```bash
uv run python app.py
# → http://localhost:7860
```

---

## Evaluation Protocol

All metrics are computed over the frozen `eval_negatives.json` pool (1 positive + 99 negatives per user). This is the standard NCF leave-one-out protocol from [He et al., 2017].

| Metric | Formula |
|---|---|
| **HR@K** | `mean_u [ i⁺_u ∈ Top-K(P_u) ]` |
| **NDCG@K** | `mean_u [ 1/log2(rank(i⁺_u) + 1) ]` if i⁺_u ∈ Top-K, else 0 |
| **Catalog Coverage@K** | `|∪_u Top-K(u)| / |I|` |
| **Personalisation@K** | `1 - (2 / |U|(|U|-1)) × Σ_{u≠v} |L_u ∩ L_v| / K` |
| **Genre Diversity@K** | `mean_u |∪_{i∈L_u} G(i)|` where G(i) is item genre set |
| **Novelty@K** | `mean_u mean_{i∈L_u} [-log2 p(i)]` where p(i) = item popularity |

---

## Optuna Search Space

| Parameter | Range | Distribution |
|---|---|---|
| `learning_rate` | [0.005, 0.3] | log-uniform |
| `num_leaves` | [20, 300] | integer |
| `max_depth` | [−1, 12] | integer |
| `min_child_samples` | [5, 100] | integer |
| `lambda_l1` | [0, 5] | uniform |
| `lambda_l2` | [0, 5] | uniform |
| `feature_fraction` | [0.5, 1.0] | uniform |
| `bagging_fraction` | [0.5, 1.0] | uniform |

The study is persisted in `checkpoints/optuna_study.db` (SQLite) — trials can be paused and resumed without loss. Best parameters are saved to `results/best_params.json`.

---

## Testing

```bash
uv run pytest                    # Full suite
uv run pytest tests/data/        # Loader + binarisation
uv run pytest tests/metrics/     # HR, NDCG, MAP, MAR
uv run pytest tests/models/      # Component tests
uv run pytest tests/integration/ # End-to-end smoke
```

---

## System Dependencies

| Package | Role |
|---|---|
| `torch` | Neural CF models, SASRec |
| `lightgbm` | LambdaRank meta-learner |
| `transformers` | Gemma base model |
| `peft` | LoRA fine-tuning |
| `accelerate` | Distributed training |
| `optuna` | HPO |
| `hydra-core` / `omegaconf` | Config management |
| `pandas` / `pyarrow` | Data manipulation + Parquet I/O |
| `numpy` | Numerical computation |
| `gradio` | Interactive UI |
| `pytest` | Testing |

---

## References

```bibtex
@inproceedings{he2017ncf,
  title={Neural Collaborative Filtering},
  author={He, Xiangnan and Liao, Lizi and Zhang, Hanwang and Nie, Liqiang and Hu, Xia and Chua, Tat-Seng},
  booktitle={WWW},
  year={2017}
}

@inproceedings{he2020lightgcn,
  title={LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={SIGIR},
  year={2020}
}

@inproceedings{rendle2009bpr,
  title={BPR: Bayesian Personalized Ranking from Implicit Feedback},
  author={Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  booktitle={UAI},
  year={2009}
}

@inproceedings{steck2019ease,
  title={Embarrassingly Shallow Autoencoders for Sparse Data},
  author={Steck, Harald},
  booktitle={WWW},
  year={2019}
}

@inproceedings{kang2018sasrec,
  title={Self-Attentive Sequential Recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={ICDM},
  year={2018}
}

@inproceedings{wang2021dcnv2,
  title={DCN V2: Improved Deep \& Cross Network},
  author={Wang, Ruoxi and Shivanna, Rakesh and Cheng, Derek and Jain, Sagar and Lin, Dong and Hong, Lichan and Chi, Ed},
  booktitle={WWW},
  year={2021}
}

@techreport{burges2010lambdamart,
  title={From RankNet to LambdaRank to LambdaMART: An Overview},
  author={Burges, Christopher J.C.},
  institution={Microsoft Research},
  year={2010}
}

@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Chen, Weizhu},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{akiba2019optuna,
  title={Optuna: A Next-generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={KDD},
  year={2019}
}

@article{harper2015movielens,
  title={The MovieLens Datasets: History and Context},
  author={Harper, F. Maxwell and Konstan, Joseph A.},
  journal={ACM Transactions on Interactive Intelligent Systems},
  volume={5},
  number={4},
  year={2015}
}
```

---

## Team

| Name | Roll No. |
|---|---|
| Nishant Tomer | 2023355 |
| Vinayak Koli | 2023597 |
| Yashasvi | 2023611 |

---

<p align="center">
  Built at IIIT-Delhi · 2025
</p>
