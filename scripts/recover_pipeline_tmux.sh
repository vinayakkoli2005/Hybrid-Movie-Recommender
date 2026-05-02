#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vinayak23597/CF-PROJECT/CF-PROJECT"
LOG_DIR="$ROOT/logs"
MASTER_LOG="$LOG_DIR/recover_pipeline.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  log "===== START: $name ====="
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  log "===== DONE: $name ====="
}

run_optional_step() {
  local name="$1"
  shift
  log "===== START OPTIONAL: $name ====="
  if "$@" 2>&1 | tee -a "$MASTER_LOG"; then
    log "===== DONE OPTIONAL: $name ====="
  else
    log "===== OPTIONAL FAILED: $name ====="
  fi
}

log "Recovery pipeline started"
log "Using python: $(which python3)"
log "Python version: $(python3 --version)"
log "PYTHONPATH=$PYTHONPATH"

run_step "Prepare Data" python3 scripts/prepare_data.py
run_step "Dump Base CF Scores" env CUDA_VISIBLE_DEVICES=0 python3 scripts/dump_scores.py
run_step "Retrain Neural Models" env CUDA_VISIBLE_DEVICES=0 python3 scripts/retrain_neural_models.py
run_step "Train SASRec" env CUDA_VISIBLE_DEVICES=0 python3 scripts/train_sasrec.py
run_optional_step "Build LLM Features (LoRA)" env CUDA_VISIBLE_DEVICES=1 python3 scripts/build_llm_features_lora.py
run_step "Train Final Model" python3 scripts/train_final_model.py
run_step "Launch UI" tmux new-window -d -t fix_pipeline -n ui_recovered "cd $ROOT && export PYTHONPATH=$ROOT/src && export PYTHONUNBUFFERED=1 && python3 app.py 2>&1 | tee -a $LOG_DIR/app_recovered.log"

log "Recovery pipeline finished"
log "Final metrics file: $ROOT/results/tuned_pipeline.json"
