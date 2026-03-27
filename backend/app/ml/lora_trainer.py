import gc
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl.trainer.sft_trainer import SFTTrainer

from app.config import settings
from app.ml.lora_data_generator import estimate_training_time, generate_training_data
from app.ml.model_manager import reset_model_manager
from app.models.repo import LoRATrainingProgress, LoRATrainingStep
from app.models.search import CodeChunk

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    pass


class _ProgressCallback(TrainerCallback):
    def __init__(
        self,
        repo_id: str,
        total_epochs: int,
        cancel_event: threading.Event,
        progress_fn: Callable[[LoRATrainingProgress], None],
    ):
        self.repo_id = repo_id
        self.total_epochs = total_epochs
        self.cancel_event = cancel_event
        self.progress_fn = progress_fn
        self._start_time = time.time()
        self._total_steps = 0

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self._total_steps = state.max_steps

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if self.cancel_event.is_set():
            raise TrainingCancelled()

        if not logs:
            return

        current_step = state.global_step
        progress = current_step / max(self._total_steps, 1)
        elapsed = time.time() - self._start_time
        eta = (
            int((elapsed / max(current_step, 1)) * (self._total_steps - current_step))
            if current_step > 0
            else None
        )

        epoch = int(state.epoch) if state.epoch else 0

        self.progress_fn(
            LoRATrainingProgress(
                repo_id=self.repo_id,
                step=LoRATrainingStep.TRAINING,
                progress=round(progress, 3),
                message=f"Step {current_step}/{self._total_steps}",
                epoch=epoch,
                total_epochs=self.total_epochs,
                train_loss=logs.get("loss"),
                eval_loss=logs.get("eval_loss"),
                estimated_time_remaining_sec=eta,
            )
        )

    def on_epoch_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self.cancel_event.is_set():
            raise TrainingCancelled()


class LoRATrainer:
    def __init__(
        self,
        repo_id: str,
        chunks: list[CodeChunk],
        progress_fn: Callable[[LoRATrainingProgress], None],
        cancel_event: threading.Event,
    ):
        self.repo_id = repo_id
        self.chunks = chunks
        self.progress_fn = progress_fn
        self.cancel_event = cancel_event

    def run(self):
        try:
            self._run_impl()
        except TrainingCancelled:
            logger.info(f"LoRA training cancelled for {self.repo_id}")
            self.progress_fn(
                LoRATrainingProgress(
                    repo_id=self.repo_id,
                    step=LoRATrainingStep.CANCELLED,
                    progress=0.0,
                    message="Training cancelled by user",
                )
            )
        except Exception as e:
            logger.exception(f"LoRA training failed for {self.repo_id} because: {e}")
            self.progress_fn(
                LoRATrainingProgress(
                    repo_id=self.repo_id,
                    step=LoRATrainingStep.FAILED,
                    progress=0.0,
                    message=f"Training failed: {str(e)}",
                )
            )

    def _run_impl(self):
        self.progress_fn(
            LoRATrainingProgress(
                repo_id=self.repo_id,
                step=LoRATrainingStep.PREPARING_DATA,
                progress=0.0,
                message="Generating training data from indexed code...",
            )
        )

        if self.cancel_event.is_set():
            raise TrainingCancelled()

        train_samples, val_samples, num_profiles = generate_training_data(self.chunks)

        if not train_samples:
            self.progress_fn(
                LoRATrainingProgress(
                    repo_id=self.repo_id,
                    step=LoRATrainingStep.FAILED,
                    progress=0.0,
                    message="Not enough data to train — need methods with meaningful bodies",
                )
            )
            return

        logger.info(
            f"Generated {len(train_samples)} training samples from {num_profiles} methods"
        )

        self.progress_fn(
            LoRATrainingProgress(
                repo_id=self.repo_id,
                step=LoRATrainingStep.PREPARING_DATA,
                progress=1.0,
                message=f"Generated {len(train_samples)} training samples from {num_profiles} methods",
            )
        )

        if self.cancel_event.is_set():
            raise TrainingCancelled()

        reset_model_manager()

        self.progress_fn(
            LoRATrainingProgress(
                repo_id=self.repo_id,
                step=LoRATrainingStep.TRAINING,
                progress=0.0,
                message="Loading base model...",
                total_epochs=settings.lora_epochs,
            )
        )

        train_ds = Dataset.from_list(
            [{"text": s["prompt"] + "\n" + s["completion"]} for s in train_samples]
        )
        val_ds = (
            Dataset.from_list(
                [{"text": s["prompt"] + "\n" + s["completion"]} for s in val_samples]
            )
            if val_samples
            else None
        )

        if len(train_ds) > 2000:
            train_ds = train_ds.shuffle(seed=42).select(range(2000))
        if val_ds and len(val_ds) > 300:
            val_ds = val_ds.shuffle(seed=42).select(range(300))

        tokenizer = AutoTokenizer.from_pretrained(
            settings.qwen_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            settings.qwen_model,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

        output_dir = settings.lora_adapters_dir / self.repo_id
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=settings.lora_epochs,
            per_device_train_batch_size=settings.lora_batch_size,
            per_device_eval_batch_size=settings.lora_batch_size,
            gradient_accumulation_steps=settings.lora_gradient_accumulation,
            learning_rate=settings.lora_lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="steps" if val_ds else "no",
            eval_steps=50 if val_ds else None,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=bool(val_ds),
            metric_for_best_model="eval_loss" if val_ds else None,
            greater_is_better=False,
            bf16=False,
            fp16=False,
            dataloader_pin_memory=False,
            report_to="none",
            max_grad_norm=1.0,
        )

        training_args.max_seq_length = 512

        progress_callback = _ProgressCallback(
            repo_id=self.repo_id,
            total_epochs=settings.lora_epochs,
            cancel_event=self.cancel_event,
            progress_fn=self.progress_fn,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            callbacks=[progress_callback],
        )

        trainer.train()

        if self.cancel_event.is_set():
            raise TrainingCancelled()

        self.progress_fn(
            LoRATrainingProgress(
                repo_id=self.repo_id,
                step=LoRATrainingStep.SAVING,
                progress=0.5,
                message="Saving LoRA adapter...",
            )
        )

        final_path = output_dir / "final"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        self._update_registry()

        self.progress_fn(
            LoRATrainingProgress(
                repo_id=self.repo_id,
                step=LoRATrainingStep.DONE,
                progress=1.0,
                message="LoRA adapter trained successfully!",
            )
        )

        logger.info(f"LoRA adapter saved to {final_path}")

    def _update_registry(self):
        registry_path = settings.indexes_dir / "registry.json"
        if not registry_path.exists():
            return

        with open(registry_path) as f:
            repos = json.load(f)

        for repo in repos:
            if repo["repo_id"] == self.repo_id:
                repo["has_lora_adapter"] = True
                break

        with open(registry_path, "w") as f:
            json.dump(repos, f, indent=2, default=str)
