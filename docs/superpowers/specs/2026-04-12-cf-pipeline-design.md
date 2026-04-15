# CF Pipeline Design — Research-Grade Collaborative Filtering on MovieLens 1M

**Date:** 2026-04-12
**Status:** Approved (brainstorming → spec phase)
**Author:** Vinay (with Claude in brainstorming mode)
**Next step:** Hand off to `superpowers:writing-plans` for the executable plan.

---

## 1. Context

This is a course project. The instructor accepts any collaborative filtering dataset, requires binarized ratings, and will grade only on a PDF report containing tables of **Hit Rate@K** and **NDCG@K** plus a YouTube link to a demo video.

The student designed an ambitious 6-stage hybrid pipeline (data → preprocessing → LLM cold start → CF retrieval → DCN ranking → LLM+RAG re-ranker → deterministic meta-learner → final ranking) and wants a research-grade execution plan that includes baselines, ablations, fine-tuning, and a robustness check.

The student has a college GPU with ~30–40 GB VRAM (A100/A6000-class), about 1+ month of focused time, and is starting greenfield (no code yet).

## 2. Goals

- Produce a reproducible, modular CF pipeline that beats standard baselines on MovieLens 1M under the standard NCF leave-one-out protocol.
- Generate **5 result tables** (headline comparison, ablation, LoRA fine-tune delta, cold-user sub-population, full-ranking robustness check).
- Produce a **5–8 minute YouTube demo** walking through architecture, live inference, and headline numbers.
- Provide a real fine-tuning result (LoRA on Llama-3.1-8B for the LLM stage), not just hyperparameter search.
- Defensible against the criticism "this might just be cherry-picked numbers" by including baselines, ablations, and a secondary full-ranking eval.

## 3. Non-Goals

- **Not** a production recommender system. No serving infra, no API, no deployment.
- **Not** a novel research contribution. Goal is to apply existing techniques rigorously.
- **Not** state-of-the-art chasing. Beating LightGCN by 0.5% NDCG is fine; the *story* matters more than the absolute number.
- **Not** going to support multiple datasets — ML-1M only, no Amazon/Yelp/Last.fm fallbacks.
- **No** MLflow/DVC/W&B mandatory infra. JSON result files in `results/` are sufficient.

## 4. Constraints (from instructor + student decisions)

| # | Constraint | Source |
|---|---|---|
| C1 | Ratings must be binarized | Problem statement |
| C2 | Metrics: Hit Rate and NDCG | Problem statement |
| C3 | Final deliverable: PDF (tables only) + YouTube link | Problem statement |
| C4 | Dataset: **MovieLens 1M** + TMDB/Kaggle metadata join | Student decision Q4 |
| C5 | Binarization: `rating ≥ 4 → 1`, all other ratings dropped | Student decision Q5 |
| C6 | LLM stages: **local Llama-3.1-8B-Instruct** via vLLM, with optional LoRA fine-tune | Student decision Q6 |
| C7 | Eval protocol: **leave-one-out + 99 sampled negatives**, K ∈ {1, 5, 10, 20} | Student decision Q7 |
| C8 | Scope: **Research-grade max** — full 6-stage pipeline + baselines + ablations + LoRA + robustness check | Student decision Q8 |
| C9 | Compute: single GPU (A100/A6000-class, 30–40 GB VRAM), ~1+ month timeline | Student environment |
| C10 | Repo style: modular Python package + Hydra configs + CLI scripts (no notebooks for production code) | Architectural decision |

## 5. Key Architectural Insight

Under the NCF leave-one-out protocol, every test user is scored against exactly **1 true positive + 99 random negatives = 100 candidate items**. This means the original pipeline's stages S1 (EASE^R + LightGCN top-200 retrieval) and S2 (DCN top-50 ranking) are **not doing retrieval** — there is nothing to retrieve from. Their role transforms from *candidate generation* to *signal generation* — each model produces a score per (user, candidate item), and these scores become **features for the meta-learner**.

This reframing is intentional and improves the design:
- Cleaner separation of concerns: every model is a feature producer; the meta-learner is the only ranker.
- More defensible: the meta-learner is deterministic (constraint from the original pipeline footer: "LLM is never the final decision maker").
- Easier ablation: dropping a stage = removing one column from the feature matrix.

## 6. Architecture

### 6.1 High-level data flow (per test user, under NCF protocol)

```
For each test user u (6,040 users on ML-1M):
    candidate_set = [u.held_out_positive] + 99 sampled_negatives  → 100 items

    For each item i in candidate_set:
        features[i] = {
            ease_score:       EASE^R(u, i),
            lgcn_score:       LightGCN(u, i),
            dcn_score:        DCN(u, i),
            dcn_uncertainty:  MC-dropout variance from DCN,
            llm_yes_prob:     P(YES | history(u), item_meta(i))     # S3
            llm_logp:         log-probability of "YES" token,
            popularity:       global interaction count of i,
            is_cold_user:     1 if |history(u)| < 5 else 0
        }
        # Cold-start LLM profile (S0) only computed if is_cold_user=1;
        # used to boost ease_score / lgcn_score for matching items.

    final_scores = meta_learner.predict(features)   # one float per item
    ranked = sort(candidate_set by final_scores, descending)

    # Metric contribution:
    rank_of_positive = ranked.index(u.held_out_positive) + 1
    HR@K  contribution: 1 if rank_of_positive ≤ K else 0
    NDCG@K contribution: 1/log2(rank_of_positive + 1) if ≤ K else 0

Aggregate: HR@K = mean over users, NDCG@K = mean over users.
```

### 6.2 Stage definitions

**S0 — LLM cold-start profile (gated)**
- Trigger: `is_cold_user = (|train_history(u)| < 5)`
- Input: user's tiny history + TMDB metadata (titles, genres, plot snippets)
- Prompt → Llama-3.1-8B → strict JSON: `{liked_genres: [...], liked_actors: [...], mood: str}`
- Output: synthetic preference vector that boosts EASE^R/LightGCN scores for matching items
- Reported separately in **Table 4** (cold-user sub-population)

**S1a — EASE^R**
- Closed-form: `B = (X^T X + λI)^-1 X^T X` then zero the diagonal of `B`
- Hyperparameter: `λ` ∈ {100, 250, 500, 1000} (tuned on validation)
- Trains in minutes on CPU; no GPU needed

**S1b — LightGCN**
- PyTorch implementation, BPR pairwise loss
- Hyperparameters: layers ∈ {1, 2, 3, 4}, embedding dim ∈ {32, 64, 128}
- ~1–3 hours on A100 for full train

**S2 — DCN-v2 with MC dropout**
- Cross network + deep MLP, dropout enabled at inference for uncertainty
- Features: user/item embeddings, popularity, EASE^R/LightGCN scores (stage cascading)
- Loss: BPR pairwise
- Output: ranking score + variance over `N=20` MC dropout passes
- Hyperparameters: cross layers ∈ {2, 3, 4}, deep widths ∈ {[256,128], [512,256,128]}, dropout ∈ {0.1, 0.3, 0.5}

**S3 — LLM + RAG decision**
- For each (user, candidate item), retrieve top-5 most similar items from user's training history:
  - **Dense retrieval**: sentence-transformers embedding of `title + genres + plot` → FAISS index
  - **Sparse retrieval**: BM25 over the same text
  - **HyDE**: LLM generates a synthetic "items this user would like" query → embed → also FAISS
  - Reciprocal rank fusion of all three → top-5
- Construct prompt: `Given user's recent likes [retrieved], would they like [candidate]? Answer strict JSON {decision: YES|NO}`
- Llama-3.1-8B emits structured output; capture token logprob of YES
- Two variants ablated: **(a) zero-shot**, **(b) LoRA-tuned** (Table 3)

**S4 — Deterministic meta-learner**
- Input feature matrix: `[ease, lgcn, dcn_score, dcn_unc, llm_yes_prob, llm_logp, popularity, is_cold]`
- Three candidate models: Logistic Regression, LightGBM, 2-layer MLP
- Best one chosen by validation NDCG@10
- Trained on validation set with the same leave-one-out structure

### 6.3 Repo structure

```
cf-project/
├── data/
│   ├── raw/                       # ml-1m.zip, tmdb_metadata.csv (gitignored)
│   ├── processed/                 # binarized.parquet, train/val/test splits
│   └── eval_negatives.json        # frozen 99-negatives per user (seeded)
├── src/cf_pipeline/
│   ├── data/                      # loaders, binarization, joins, splits
│   │   ├── loaders.py
│   │   ├── binarize.py
│   │   ├── join_tmdb.py
│   │   └── splits.py
│   ├── models/
│   │   ├── baselines.py           # Popularity, ItemKNN, BPR-MF, NeuMF
│   │   ├── ease.py
│   │   ├── lightgcn.py
│   │   ├── dcn.py
│   │   └── meta.py                # LR / LightGBM / MLP meta-learner
│   ├── llm/
│   │   ├── server.py              # vLLM wrapper (offline batched inference)
│   │   ├── cold_start.py          # S0 prompt + parser
│   │   ├── rag.py                 # FAISS + BM25 + HyDE
│   │   ├── decision.py            # S3 strict-JSON YES/NO + logprob extraction
│   │   └── lora_finetune.py       # LoRA training script (PEFT)
│   ├── eval/
│   │   ├── metrics.py             # HR@K, NDCG@K (vectorized)
│   │   ├── protocol.py            # leave-one-out, sampled negatives
│   │   └── full_ranking.py        # secondary robustness eval
│   ├── pipeline/
│   │   ├── inference.py           # end-to-end per-user pipeline
│   │   └── features.py            # feature stacking for meta-learner
│   └── utils/
│       ├── seeds.py
│       ├── logging.py
│       └── io.py
├── configs/                       # Hydra YAMLs
│   ├── data/ml1m.yaml
│   ├── model/{ease, lightgcn, dcn, neumf, ...}.yaml
│   ├── llm/{llama8b_zeroshot, llama8b_lora}.yaml
│   ├── eval/{ncf_protocol, full_ranking}.yaml
│   └── experiment/{baseline_pop, baseline_itemknn, headline, ablate_no_llm, ...}.yaml
├── scripts/
│   ├── prepare_data.py            # one-shot: download, binarize, split, freeze negatives
│   ├── train.py                   # hydra entrypoint: train any model from config
│   ├── eval.py                    # hydra entrypoint: eval any model from config
│   ├── run_pipeline.py            # full S0→S4 inference + per-user feature dump
│   ├── lora_train.py              # LoRA finetune Llama on (history → liked) pairs
│   └── make_tables.py             # collate results/*.json → markdown/LaTeX tables
├── results/
│   ├── tableN_*.json              # per-experiment outputs
│   └── tables.md                  # final auto-generated report tables
├── notebooks/
│   └── 00_exploration.ipynb       # EDA only — never on the critical path
├── report/
│   ├── report.tex
│   └── report.pdf
├── tests/                         # pytest: metrics correctness, splits, leakage checks
│   ├── test_metrics.py
│   ├── test_splits.py
│   └── test_data_leakage.py
├── pyproject.toml                 # uv-managed
├── README.md
└── docs/superpowers/specs/2026-04-12-cf-pipeline-design.md   # this file
```

## 7. Data Design

### 7.1 Sources
- **MovieLens 1M**: 1,000,209 ratings, 6,040 users, 3,706 items, ratings 1–5, timestamps
- **MovieLens links.csv**: maps `movieId` → `imdbId`, `tmdbId`
- **TMDB Kaggle metadata**: title, overview, genres, cast, keywords, popularity (joined on `tmdbId`)

### 7.2 Preprocessing pipeline (`scripts/prepare_data.py`)
1. Download ML-1M and TMDB metadata
2. Join ratings with metadata via `links.csv` → drop items with no metadata match (`<3% loss expected`)
3. **Binarize**: `rating ≥ 4 → keep as positive`; `rating < 4 → drop` (positive-only training set)
4. **Split (leave-one-out)**: for each user, sort by timestamp; latest interaction → test, second-latest → validation, rest → training
5. **Freeze negatives**: for each (user, test_positive), sample 99 items the user has never interacted with using `np.random.default_rng(42)`; persist to `data/eval_negatives.json`
6. Same negative-sampling for validation set (separate file)

### 7.3 Outputs
- `data/processed/train.parquet` — `(user_id, item_id, timestamp)`
- `data/processed/val.parquet` — `(user_id, val_positive, [99 negatives])`
- `data/processed/test.parquet` — `(user_id, test_positive, [99 negatives])`
- `data/processed/items_metadata.parquet` — item-level features for LLM/RAG
- All splits committed (small) so every experiment is reproducible

## 8. Evaluation Protocol

### 8.1 Primary protocol (NCF leave-one-out + 99 negatives)
- For each test user: rank `[positive] + 99_negatives = 100 items`
- Compute `HR@K` and `NDCG@K` for `K ∈ {1, 5, 10, 20}`
- Aggregate by mean across users (with std-dev across 5 different random seeds for negative resampling — only for headline model)
- All 8 metrics reported in every table

### 8.2 Secondary protocol (Table 5 only)
- Same leave-one-out hold-out, but rank against ALL items the user has not interacted with (~3,700 unseen per user)
- Only run for the headline pipeline (and a Popularity baseline as a sanity floor)
- Justification: shuts down "99-negatives is too easy" criticism

### 8.3 Implementation notes
- Vectorized scoring: stack candidate items into a (n_users, 100) tensor, do ONE forward pass per model
- A `BaseRanker` interface: `score(user_ids, candidate_item_ids) → (n_users, 100) score matrix`. Every model implements this.
- `eval_pipeline(model)` is a single function that produces all 8 metrics in one call

## 9. Fine-Tuning Scope

| Component | Search space | Method |
|---|---|---|
| EASE^R | λ ∈ {100, 250, 500, 1000} | Grid on val NDCG@10 |
| LightGCN | layers ∈ {1,2,3,4}, dim ∈ {32,64,128}, lr ∈ {1e-3, 5e-4} | Grid on val NDCG@10 |
| DCN-v2 | cross_layers ∈ {2,3,4}, deep ∈ {[256,128],[512,256,128]}, dropout ∈ {0.1,0.3,0.5} | Optuna, 30 trials |
| LLM (LoRA) | rank ∈ {8,16,32}, alpha ∈ {16,32}, target = `q_proj,v_proj`, 3 epochs on ~50K (history → liked) pairs | Single best config from quick sweep |
| Meta-learner | LR vs LightGBM vs MLP, plus per-model hyperparam search | Optuna, 50 trials |

LoRA training corpus: from the **training split only**, generate `(user history snippet → "YES" label, sampled negative item → "NO" label)` pairs. ~50K total. Uses HuggingFace PEFT + Transformers.

## 10. Reproducibility

- Single global seed (`SEED=42`), set via `cf_pipeline.utils.seeds.set_global_seed()` propagating to: `random`, `numpy`, `torch`, `torch.cuda`, `transformers`, `np.random.default_rng`
- Frozen `data/eval_negatives.json` committed
- `pyproject.toml` with pinned versions, uv lockfile committed
- Each `results/*.json` records: `git_sha`, `config_hash`, `seed`, `timestamp`, `metrics`, `runtime_seconds`
- `make_tables.py` is deterministic — same JSON inputs always produce the same markdown table

## 11. Deliverables

### 11.1 Result tables (auto-generated → embedded in report PDF)

| # | Table | Rows | Columns |
|---|---|---|---|
| 1 | Headline | Pop, ItemKNN, BPR-MF, NeuMF, EASE^R, LightGCN, DCN, **Full pipeline** | HR@{1,5,10,20}, NDCG@{1,5,10,20} |
| 2 | Ablation | Full minus {S0, EASE^R, LightGCN, DCN, LLM-RAG, meta-learner} | same |
| 3 | LoRA delta | Pipeline w/ zero-shot Llama vs. LoRA-tuned | same |
| 4 | Cold-user sub-population | Headline, with vs. without S0, on users with <5 train interactions | same |
| 5 | Full-ranking robustness | Headline pipeline under full ranking + Popularity baseline | HR@{10,20}, NDCG@{10,20} |

### 11.2 PDF report
- ~3–5 pages
- Sections: 1-paragraph context, 1-paragraph data/protocol, 5 tables, 1-paragraph conclusion, YouTube link
- LaTeX (`report/report.tex`) so tables are crisp

### 11.3 YouTube demo video (5–8 minutes)
- 1 min: problem + dataset + binarization + protocol
- 2 min: pipeline architecture walkthrough (using the SVG diagram)
- 2 min: live inference on one sample user (show top-K and held-out positive position)
- 1 min: tour of the 5 tables and headline number
- 1 min: conclusions and what surprised you
- Recorded with screen capture; uploaded unlisted to YouTube

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM stage too slow at eval (100 items × 6,040 users = 604K calls) | High | High | vLLM batched inference; cache prompts; only run S3 on top-50 items per user (cheap pre-filter via DCN) |
| LightGCN doesn't beat EASE^R on ML-1M | Medium | Low | This is a known finding in the literature; report it honestly as part of the story |
| LoRA fine-tune doesn't improve over zero-shot | Medium | Medium | Negative result is still a result; report it. Hyperparameter sweep mitigates somewhat. |
| Meta-learner overfits the validation set (only 6,040 examples) | Medium | Medium | Use cross-validation on the validation positives; LR is naturally regularized |
| Data leakage between train and test | Low | Catastrophic | `tests/test_data_leakage.py` runs in CI; asserts no test items appear in any user's training history |
| Cold-user S0 stage is too sparse to evaluate (very few users with <5 interactions in ML-1M) | Medium | Low | Lower the threshold to <10 if needed; report n in Table 4 |
| Time overrun on the full ranking eval (Table 5) | Low | Low | Run only for headline + Popularity; sample 1000 users if needed |

## 13. Success Criteria

- [ ] All 5 tables filled with real numbers
- [ ] Headline pipeline beats every baseline by ≥1% absolute on NDCG@10
- [ ] Ablation shows every stage contributes ≥0.5% absolute on NDCG@10 (or is honestly reported as not contributing)
- [ ] LoRA delta shows a measurable change vs zero-shot (sign matters more than magnitude)
- [ ] Full-ranking robustness check shows headline pipeline is still best (relative ordering preserved)
- [ ] All experiments reproducible from a single `make all` (or equivalent script chain)
- [ ] PDF + YouTube link submitted before deadline

## 14. Open Questions / To Confirm in Writing-Plans Phase

- Exact hyperparameters for the LightGCN/DCN search grids (defaults proposed; refine in plan)
- Whether to use Optuna or simple grid for fine-tuning (Optuna preferred — confirm in plan)
- LoRA dataset construction details (50K pair recipe — confirm in plan)
- Whether to include NeuMF as a baseline or skip (it's often dominated by LightGCN; confirm in plan)
- Whether to record per-stage runtime/memory in result JSONs (recommended; confirm in plan)

---

**End of design spec.** Hand off to `superpowers:writing-plans` for the executable plan.
