import gc
import logging
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.ml.unixcoder import UniXcoderWrapper

logger = logging.getLogger(__name__)

_manager = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelManager:
    def __init__(self, lora_adapter_path: str | None = None):
        self.device = get_device()
        self._unixcoder = None
        self._unixcoder_tokenizer = None
        self._qwen = None
        self._qwen_tokenizer = None
        self._lora_adapter_path = lora_adapter_path
        logger.info(f"MM device: {self.device}")
        if lora_adapter_path:
            logger.info(f"using LORA adapter in {lora_adapter_path}")

    @property
    def unixcoder(self):
        if self._unixcoder is None:
            self._unixcoder = UniXcoderWrapper(self.device)
            logger.info("unixcoder loaded")
        return self._unixcoder

    @property
    def qwen_tokenizer(self):
        if self._qwen_tokenizer is None:
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                settings.qwen_model, trust_remote_code=True
            )
            logger.info("qwen tokenizer loaded")
        return self._qwen_tokenizer

    @property
    def qwen(self):
        if self._qwen is None:
            use_lora = (
                self._lora_adapter_path and Path(self._lora_adapter_path).exists()
            )
            use_fp32 = use_lora or self.device == "cpu"
            dtype = torch.float32 if use_fp32 else torch.float16

            base_model = AutoModelForCausalLM.from_pretrained(
                settings.qwen_model,
                dtype=dtype,
                trust_remote_code=True,
            )

            if use_lora and self._lora_adapter_path is not None:
                logger.info(f"using LORA adapter {self._lora_adapter_path}")
                peft_model = PeftModel.from_pretrained(
                    base_model, self._lora_adapter_path
                )
                base_model = peft_model.merge_and_unload()
                del peft_model

            self._qwen = base_model.to(self.device)
            self._qwen.eval()
            logger.info(f"loaded qwen <{dtype}> <{self.device}>")
        return self._qwen

    def encode_code(self, texts: list[str]) -> np.ndarray:
        return self.unixcoder.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.unixcoder.encode([text])

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.qwen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        with torch.no_grad():
            outputs = self.qwen.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)


def get_model_manager(lora_adapter_path: str | None = None) -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager(lora_adapter_path=lora_adapter_path)
    return _manager


def reset_model_manager():
    global _manager
    if _manager is not None:
        if _manager._qwen is not None:
            del _manager._qwen
        if _manager._unixcoder is not None:
            del _manager._unixcoder
        del _manager
        _manager = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        gc.collect()
