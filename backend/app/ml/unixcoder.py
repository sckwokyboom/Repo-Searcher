import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings


class UniXcoderWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(settings.unixcoder_model)
        self.model = AutoModel.from_pretrained(settings.unixcoder_model).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 512) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()
