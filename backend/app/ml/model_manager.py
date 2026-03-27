import gc
import logging
import threading
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.ml.unixcoder import UniXcoderWrapper

logger = logging.getLogger(__name__)

_manager = None
_manager_lock = threading.Lock()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelManager:
    def __init__(self, lora_adapter_path: str | None = None):
        self.device = get_device()
        self._qwen = None
        self._qwen_tokenizer = None
        self._lora_adapter_path = lora_adapter_path
        logger.info(f"MM device: {self.device}")
        if lora_adapter_path:
            logger.info(f"using LORA adapter in {lora_adapter_path}")

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

    @property
    def lora_adapter_path(self) -> str | None:
        return self._lora_adapter_path

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
    with _manager_lock:
        if _manager is None:
            _manager = ModelManager(lora_adapter_path=lora_adapter_path)
        return _manager


def ensure_lora_adapter(adapter_path: str | None) -> ModelManager:
    """Ensure the ModelManager singleton uses the given LoRA adapter."""
    global _manager
    with _manager_lock:
        current_path = _manager.lora_adapter_path if _manager else None
        requested = str(adapter_path) if adapter_path else None
        current = str(current_path) if current_path else None

        if current != requested:
            logger.info(f"Switching LoRA adapter: {current} -> {requested}")
            _reset_manager_unsafe()
            _manager = ModelManager(lora_adapter_path=requested)

        return _manager


def reset_model_manager():
    with _manager_lock:
        _reset_manager_unsafe()


def _reset_manager_unsafe():
    global _manager
    if _manager is not None:
        if _manager._qwen is not None:
            del _manager._qwen
        del _manager
        _manager = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        gc.collect()
