from __future__ import annotations

import bitsandbytes
import torch
import transformers

from cf_pipeline.utils.logging import get_logger

_log = get_logger("llm.server")


class LlamaServer:
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: str = "bfloat16",
        max_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.dtype = getattr(torch, dtype)
        self.max_tokens = max_tokens
        self._bnb = bitsandbytes
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=self.dtype,
        )
        self._generation_config = transformers.GenerationConfig(
            temperature=0.0,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        _log.info("Loaded %s", model_id)

    def generate(self, prompts: list[str]) -> list[dict]:
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                generation_config=self._generation_config,
            )

        texts = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [{"text": text, "logprobs": None} for text in texts]

    def free(self) -> None:
        del self._model
        del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
