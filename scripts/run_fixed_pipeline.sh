#!/bin/bash
set -e
cd /home/vinayak23597/CF-PROJECT/CF-PROJECT
LOG=logs/fixed_pipeline.log

echo "[$(date)] === Step 1: Train SASRec (fixed NaN bug) ===" | tee -a $LOG
python3.8 scripts/train_sasrec.py 2>&1 | tee -a $LOG

echo "[$(date)] === Step 2: Re-tune meta-learner (100 trials) ===" | tee -a $LOG
python3.8 scripts/tune_meta_learner.py 2>&1 | tee -a $LOG

echo "[$(date)] === Step 3: Train final model ===" | tee -a $LOG
python3.8 scripts/train_final_model.py 2>&1 | tee -a $LOG

echo "[$(date)] === Step 4: Generate tables ===" | tee -a $LOG
python3.8 scripts/generate_tables.py 2>&1 | tee -a $LOG

echo "[$(date)] === ALL DONE ===" | tee -a $LOG
