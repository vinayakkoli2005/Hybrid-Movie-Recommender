from __future__ import annotations

import pytest
import torch

from cf_pipeline.llm.server import LlamaServer


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_basic():
    srv = LlamaServer()
    out = srv.generate(["Reply with the word HELLO and nothing else."])
    assert "HELLO" in out[0]["text"].upper()
    srv.free()
