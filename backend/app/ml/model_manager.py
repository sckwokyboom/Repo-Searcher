import logging
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings

logger = logging.getLogger(__name__)

_manager = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelManager:
    def __init__(self):
        self.device = get_device()
        self._unixcoder = None
        self._unixcoder_tokenizer = None
        self._qwen = None
        self._qwen_tokenizer = None
        logger.info(f"ModelManager initialized with device: {self.device}")

    @property
    def unixcoder(self):
        if self._unixcoder is None:
            from app.ml.unixcoder import UniXcoderWrapper

            self._unixcoder = UniXcoderWrapper(self.device)
            logger.info("UniXcoder loaded")
        return self._unixcoder

    @property
    def qwen_tokenizer(self):
        if self._qwen_tokenizer is None:
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                settings.qwen_model, trust_remote_code=True
            )
            logger.info("Qwen tokenizer loaded")
        return self._qwen_tokenizer

    @property
    def qwen(self):
        if self._qwen is None:
            self._qwen = AutoModelForCausalLM.from_pretrained(
                settings.qwen_model,
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self._qwen.eval()
            logger.info("Qwen model loaded")
        return self._qwen

    def encode_code(self, texts: list[str]) -> np.ndarray:
        return self.unixcoder.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.unixcoder.encode([text])

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.qwen_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.qwen.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)


def get_model_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
