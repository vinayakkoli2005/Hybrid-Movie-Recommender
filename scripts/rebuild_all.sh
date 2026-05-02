#!/usr/bin/env bash
# Full pipeline rebuild: download → prepare → CF models → SASRec → tune → final → UI
set -e
ROOT="/home/vinayak23597/CF-PROJECT/CF-PROJECT"
LOG="$ROOT/logs/rebuild.log"
mkdir -p "$ROOT/logs" "$ROOT/data/raw/ml-1m" "$ROOT/data/raw/tmdb" "$ROOT/data/processed" "$ROOT/results" "$ROOT/checkpoints"
cd "$ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
log "=========================================="
log " FULL PIPELINE REBUILD STARTED"
log "=========================================="

# ── 1. ML-1M ratings + movies ────────────────────────────────────────────────
log "STEP 1: Download ML-1M"
if [ ! -f "data/raw/ml-1m/ratings.dat" ]; then
    wget -q --show-progress \
        https://files.grouplens.org/datasets/movielens/ml-1m.zip \
        -O /tmp/ml-1m.zip 2>&1 | tee -a "$LOG"
    unzip -o /tmp/ml-1m.zip -d /tmp/ 2>&1 | tee -a "$LOG"
    cp /tmp/ml-1m/ratings.dat data/raw/ml-1m/
    cp /tmp/ml-1m/movies.dat  data/raw/ml-1m/
    log "  ML-1M done."
else
    log "  Already exists, skipping."
fi

# ── 2. links.csv from ML-25M ─────────────────────────────────────────────────
log "STEP 2: Download links.csv (ML-25M)"
if [ ! -f "data/raw/ml-1m/links.csv" ]; then
    wget -q --show-progress \
        https://files.grouplens.org/datasets/movielens/ml-25m.zip \
        -O /tmp/ml-25m.zip 2>&1 | tee -a "$LOG"
    unzip -o /tmp/ml-25m.zip ml-25m/links.csv -d /tmp/ 2>&1 | tee -a "$LOG"
    cp /tmp/ml-25m/links.csv data/raw/ml-1m/
    log "  links.csv done."
else
    log "  Already exists, skipping."
fi

# ── 3. TMDB metadata via Kaggle ───────────────────────────────────────────────
log "STEP 3: Download TMDB metadata (Kaggle)"
if [ ! -f "data/raw/tmdb/movies_metadata.csv" ]; then
    mkdir -p ~/.kaggle
    echo '{"username":"vinayak23597","key":"KGAT_07e26aeec0dfdf74d5fed6ec2f9e9fe6"}' > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    pip install -q kaggle 2>&1 | tee -a "$LOG"
    python3.8 -m kaggle datasets download -d rounakbanik/the-movies-dataset \
        -p /tmp/tmdb/ --unzip 2>&1 | tee -a "$LOG"
    cp /tmp/tmdb/movies_metadata.csv data/raw/tmdb/
    log "  TMDB done."
else
    log "  Already exists, skipping."
fi

# ── 4. Prepare data ───────────────────────────────────────────────────────────
log "STEP 4: Prepare data (split + negatives + metadata)"
python3.8 scripts/prepare_data.py 2>&1 | tee -a "$LOG"
log "  Data prepared."

# ── 5. Train base CF models (pop, knn, ease, bpr, lgcn, dcn, neumf) ──────────
log "STEP 5: Train base CF models"
python3.8 scripts/dump_scores.py 2>&1 | tee -a "$LOG"
log "  Base CF models done."

# ── 6. Retrain neural models with full epochs ─────────────────────────────────
log "STEP 6: Retrain neural models (LightGCN, DCN, NeuMF)"
python3.8 scripts/retrain_neural_models.py 2>&1 | tee -a "$LOG"
log "  Neural models done."

# ── 7. Train SASRec ───────────────────────────────────────────────────────────
log "STEP 7: Train SASRec (200 epochs)"
python3.8 scripts/train_sasrec.py 2>&1 | tee -a "$LOG"
log "  SASRec done."

# ── 8. Tune meta-learner ──────────────────────────────────────────────────────
log "STEP 8: Tune LambdaRank meta-learner (Optuna 80 trials)"
python3.8 scripts/tune_meta_learner.py 2>&1 | tee -a "$LOG"
log "  Tuning done."

# ── 9. Train final model + evaluate ───────────────────────────────────────────
log "STEP 9: Train final model + evaluate on test set"
python3.8 scripts/train_final_model.py 2>&1 | tee -a "$LOG"
log "  Final model done."

# ── 10. Print results ─────────────────────────────────────────────────────────
log "=========================================="
log " FINAL RESULTS"
log "=========================================="
python3.8 - 2>&1 | tee -a "$LOG" <<'PYEOF'
import json
from pathlib import Path
r = json.loads(Path("results/tuned_pipeline.json").read_text())
m = r["metrics"]
print()
print("="*52)
print("  FINAL TEST SET RESULTS")
print("="*52)
for k in [1, 5, 10, 20]:
    print(f"  HR@{k:<2} = {m[f'HR@{k}']:.4f}  "
          f"NDCG@{k:<2} = {m[f'NDCG@{k}']:.4f}  "
          f"MAP@{k:<2} = {m[f'MAP@{k}']:.4f}  "
          f"Novelty@{k:<2} = {m.get(f'Novelty@{k}', 0):.4f}")
print("="*52)
print(f"  Best val NDCG@10 = {r['best_val_ndcg']:.4f}")
print("="*52)
PYEOF

# ── 11. Launch UI ─────────────────────────────────────────────────────────────
log "STEP 11: Launching Gradio UI on :7860"
python3.8 app.py 2>&1 | tee logs/app.log
