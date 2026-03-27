import gc
import json
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GENERATOR_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

_model = None
_tokenizer = None
_device = None


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str = GENERATOR_MODEL) -> tuple:
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _device = "cpu"
    print(f"Loading {model_name} on {_device} (float32)...")
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    ).eval()
    param_count = sum(p.numel() for p in _model.parameters()) / 1e9
    print(f"Loaded {param_count:.1f}B params on {_device}")

    return _model, _tokenizer, _device


def unload_model():
    global _model, _tokenizer, _device
    del _model, _tokenizer
    _model = None
    _tokenizer = None
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    print("Generator model unloaded")


def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    model_name: str = GENERATOR_MODEL,
) -> str:
    model, tokenizer, device = load_model(model_name)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError, Exception:
        if system:
            text = f"{system}\n\n{prompt}"
        else:
            text = prompt

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(
        device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if "<think>" in response:
        response = re.sub(
            r"<think>.*?</think>\s*", "", response, flags=re.DOTALL
        ).strip()

    return response


def generate_json_array(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> list[str] | None:
    raw = generate(
        prompt, system=system, temperature=temperature, max_new_tokens=max_new_tokens
    )

    text = raw.strip()

    if "```" in text:
        for part in text.split("```")[1:]:
            c = part.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("["):
                text = c
                break

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            return data
    except json.JSONDecodeError, TypeError:
        pass

    return None


def batch_generate_json_arrays(
    prompts: list[str],
    system: str = "",
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    cache_path: Path | None = None,
    progress_every: int = 10,
) -> list[list[str] | None]:
    cache: dict[str, list[str]] = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached responses from {cache_path}")

    results: list[list[str] | None] = []
    new_count = 0
    fail_count = 0

    for i, prompt in enumerate(prompts):
        cache_key = str(hash(prompt))

        if cache_key in cache:
            results.append(cache[cache_key])
        else:
            parsed = generate_json_array(
                prompt,
                system=system,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            results.append(parsed)
            if parsed is not None:
                cache[cache_key] = parsed
                new_count += 1
            else:
                fail_count += 1

        if (i + 1) % progress_every == 0 or i == len(prompts) - 1:
            print(
                f"[{i + 1}/{len(prompts)}] "
                f"{new_count} new, {i + 1 - new_count - fail_count} cached, "
                f"{fail_count} failed"
            )

    if cache_path and new_count > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"Saved {len(cache)} entries to cache {cache_path}")

    return results


def test_connectivity(model_name: str = GENERATOR_MODEL) -> bool:
    try:
        result = generate(
            "Generate 2 code search queries for a Java validation method. "
            "Output as a JSON array of strings.",
            system="Output only a JSON array of strings.",
            max_new_tokens=100,
        )
        print(result)
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    ok = test_connectivity()
    if ok:
        result = generate_json_array(
            "Generate 3 code search queries for a Java method that validates user input fields.",
            system="You generate realistic code search queries. Output only a JSON array of strings.",
        )
        unload_model()
