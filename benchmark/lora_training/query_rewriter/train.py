"""
LoRA fine-tuning of Qwen2.5-Coder-1.5B for query rewriting.

Trains the model to transform natural language queries into structured
retrieval-oriented JSON with project-specific terms and search hints.
Uses PEFT LoRA for parameter-efficient training on Apple Silicon (MPS).

Usage:
  python train.py              # train on v3 data (default)
  python train.py --version v2 # train on v2 data
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

DATA_DIR = Path(__file__).parent / "data"
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"

# Version-specific paths
VERSION_CONFIG = {
    "v2": {
        "train": "train_rewriter_v2.jsonl",
        "val": "val_rewriter_v2.jsonl",
        "output": "rewriter_lora_v2",
        "epochs": 3,
    },
    "v3": {
        "train": "train_rewriter_v3.jsonl",
        "val": "val_rewriter_v3.jsonl",
        "output": "rewriter_lora_v3",
        "epochs": 5,
    },
    "v4": {
        "train": "train_rewriter_v4.jsonl",
        "val": "val_rewriter_v4.jsonl",
        "output": "rewriter_lora_v4",
        "epochs": 5,
    },
}


def load_data(path: Path) -> Dataset:
    """Load JSONL training data into HuggingFace Dataset."""
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            text = obj["prompt"] + "\n" + obj["completion"]
            samples.append({"text": text})
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3", choices=list(VERSION_CONFIG.keys()))
    args = parser.parse_args()

    cfg = VERSION_CONFIG[args.version]
    output_dir = Path(__file__).parent.parent / "output" / cfg["output"]
    train_path = DATA_DIR / cfg["train"]
    val_path = DATA_DIR / cfg["val"]
    num_epochs = cfg["epochs"]

    print(f"Version: {args.version}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Training data: {train_path}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (float16 on MPS)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, trust_remote_code=True,
    ).to("mps")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    MAX_TRAIN = 2000
    MAX_VAL = 300
    print("Loading datasets...")
    train_dataset = load_data(train_path)
    val_dataset = load_data(val_path)
    if len(train_dataset) > MAX_TRAIN:
        train_dataset = train_dataset.shuffle(seed=42).select(range(MAX_TRAIN))
    if len(val_dataset) > MAX_VAL:
        val_dataset = val_dataset.shuffle(seed=42).select(range(MAX_VAL))
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        max_grad_norm=1.0,
    )

    training_args.max_seq_length = 512
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAdapter saved to {final_path}")

    print("\nEvaluating...")
    metrics = trainer.evaluate()
    print(f"Eval loss: {metrics['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
