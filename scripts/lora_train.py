"""Fine-tune Gemma-2-9b-it with LoRA on YES/NO recommendation decisions.

LoRA config: q_proj + v_proj, rank=16, alpha=32, dropout=0.05
Training: 2 epochs, bf16, gradient checkpointing, single A100-40GB.

Run:
  CUDA_VISIBLE_DEVICES=0 python3.8 scripts/lora_train.py 2>&1 | tee logs/lora_train.log
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

from cf_pipeline.utils.logging import get_logger

MODEL_ID = "models/gemma-2-9b-it"
DATA_PATH = Path("data/processed/lora_train.jsonl")
OUT_DIR = Path("checkpoints/lora")
MAX_LEN = 512
LOG_STEPS = 50
SAVE_STEPS = 200


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


class DecisionDataset(Dataset):
    """Tokenizes (prompt, response) pairs with prompt-token masking."""

    def __init__(self, records: list[dict], tokenizer) -> None:
        self.tokenizer = tokenizer
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        tok = self.tokenizer

        # Full conversation: user prompt + model response
        full_text = tok.apply_chat_template(
            [
                {"role": "user", "content": rec["prompt"]},
                {"role": "model", "content": rec["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Prompt only (to find where response starts)
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": rec["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_enc = tok(
            full_text,
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        prompt_enc = tok(
            prompt_text,
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]

        labels = input_ids.clone()
        # Mask all prompt tokens — loss only on response tokens
        prompt_len = min(prompt_enc["input_ids"].shape[1], len(labels))
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main() -> None:
    log = get_logger("lora_train")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer from %s…", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model in bfloat16…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    # Reset generation config (avoids Gemma-2 HybridCache stale state)
    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.enable_input_require_grads()

    # Apply LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    log.info("Loading dataset from %s…", DATA_PATH)
    records = _load_jsonl(DATA_PATH)
    # 90/10 train/val split
    split = int(0.9 * len(records))
    train_ds = DecisionDataset(records[:split], tokenizer)
    val_ds = DecisionDataset(records[split:], tokenizer)
    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    # DataCollator pads variable-length sequences and handles label masking
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,       # effective batch = 32
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    log.info("Starting LoRA fine-tuning…")
    trainer.train()

    log.info("Saving adapter to %s…", OUT_DIR / "final")
    model.save_pretrained(str(OUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUT_DIR / "final"))
    log.info("Done.")


if __name__ == "__main__":
    main()
