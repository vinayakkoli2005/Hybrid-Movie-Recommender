from __future__ import annotations

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from cf_pipeline.utils.logging import get_logger

_log = get_logger("llm.server")

# Default: local copy in project folder
_DEFAULT_MODEL = "models/gemma-2-9b-it"


class LlamaServer:
    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        dtype: str = "bfloat16",
        max_tokens: int = 256,
        device: str = "cuda:0",
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.device = device
        torch_dtype = getattr(torch, dtype)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model onto a single GPU — device_map="auto" splits across GPUs
        # and causes Gemma-2 to produce garbage output (HybridCache bug)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(device)
        self._model.eval()

        # Reset generation config to avoid HybridCache from old checkpoint
        self._model.generation_config = GenerationConfig(
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        _log.info("Loaded %s on %s in %s", model_id, device, dtype)

    def generate(self, prompts: list[str]) -> list[dict]:
        # Apply chat template to every prompt
        texts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        # Left-pad so all sequences align on the right (required for causal LM batching)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        results = []
        for i in range(len(prompts)):
            new_tokens = outputs[i][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append({"text": text, "logprobs": None})
        return results

    def free(self) -> None:
        del self._model
        del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
