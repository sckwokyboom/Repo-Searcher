"""
LoRA fine-tuning of Qwen2.5-Coder-1.5B for code relevance scoring.

Trains the model to output a relevance score (0-10) given a query + code context.
Uses PEFT LoRA for parameter-efficient training on Apple Silicon (MPS).
"""

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output" / "scorer_lora"
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"


def load_data(path: Path) -> Dataset:
    """Load JSONL training data into HuggingFace Dataset."""
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            # Format as a single text for SFT: prompt + completion
            text = obj["prompt"] + " " + obj["completion"]
            samples.append({"text": text})
    return Dataset.from_list(samples)


def main():
    print(f"Base model: {BASE_MODEL}")
    print(f"Training data: {DATA_DIR / 'train_scorer.jsonl'}")
    print(f"Output: {OUTPUT_DIR}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model on MPS with float16
    print("Loading base model (float16 on MPS)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, trust_remote_code=True,
    ).to("mps")

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets (cap size for faster training on MPS)
    MAX_TRAIN = 1000
    MAX_VAL = 200
    print("Loading datasets...")
    train_dataset = load_data(DATA_DIR / "train_scorer.jsonl")
    val_dataset = load_data(DATA_DIR / "val_scorer.jsonl")
    if len(train_dataset) > MAX_TRAIN:
        train_dataset = train_dataset.shuffle(seed=42).select(range(MAX_TRAIN))
    if len(val_dataset) > MAX_VAL:
        val_dataset = val_dataset.shuffle(seed=42).select(range(MAX_VAL))
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=False,  # model already in float16, MPS handles it
        dataloader_pin_memory=False,
        report_to="none",
        max_grad_norm=1.0,
    )

    # Trainer
    training_args.max_seq_length = 256
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final adapter
    final_path = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAdapter saved to {final_path}")

    # Quick eval
    print("\nEvaluating...")
    metrics = trainer.evaluate()
    print(f"Eval loss: {metrics['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
