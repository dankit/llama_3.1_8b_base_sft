"""Fine-tune LLM with LoRA using Unsloth."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unsloth must be imported before trl/transformers for optimizations
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import json
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig

from config import (
    BASE_MODEL, MAX_SEQ_LENGTH, PRECISION, get_precision_kwargs,
    TRAINING_OUTPUT_DIR, LORA_OUTPUT_DIR, LORA_CONFIG, CHAT_TEMPLATE,
    DATASET_NAME, DATASET_SPLIT, DATASET_NUM_SAMPLES,
)

WANDB_PROJECT = "llama-ifeval-finetuning"

# Optimized for Llama-3.1-8B + Alpaca-cleaned (52k) + LoRA bf16 on A100-40GB
TRAINING_CONFIG = {
    "output_dir": TRAINING_OUTPUT_DIR,
    "per_device_train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.03,
    "num_train_epochs": 1,
    "learning_rate": 5e-4,
    "logging_steps": 1,
    "optim": "adamw_torch_fused",  # Fast fused CUDA kernels
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "bf16": True,
    "fp16": False,
    "tf32": True,  # A100 tensor cores
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "save_strategy": "epoch",
    "report_to": "wandb",
}


def format_alpaca(examples, tokenizer):
    """Format alpaca-cleaned dataset into Llama-3.1 chat format."""
    texts = []
    for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"]):
        user = f"{instruction}\n\n{inp}" if inp and inp.strip() else instruction
        conv = [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
        texts.append(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))
    return {"text": texts}


def main():
    run_name = f"train_{PRECISION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=WANDB_PROJECT, name=run_name, config={
        "base_model": BASE_MODEL, "precision": PRECISION, "max_seq_length": MAX_SEQ_LENGTH,
        "dataset": DATASET_NAME, "lora_r": LORA_CONFIG["r"], "lora_alpha": LORA_CONFIG["lora_alpha"],
        **TRAINING_CONFIG,
    }, tags=["training", PRECISION])

    print(f"Loading {BASE_MODEL} ({PRECISION})...")
    prec = get_precision_kwargs()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        dtype=prec["dtype"], load_in_4bit=prec["load_in_4bit"], load_in_8bit=prec["load_in_8bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    print(f"Adding LoRA adapters (r={LORA_CONFIG['r']})...")
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    trainable, total = sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    if DATASET_NUM_SAMPLES:
        dataset = dataset.select(range(min(DATASET_NUM_SAMPLES, len(dataset))))
    dataset = dataset.map(lambda x: format_alpaca(x, tokenizer), batched=True, remove_columns=dataset.column_names)
    print(f"Dataset: {len(dataset)} samples")

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            **TRAINING_CONFIG,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,
        ),
    )

    print("Training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stats = trainer.train()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"Done! Steps: {stats.global_step}, Loss: {stats.training_loss:.4f}, Peak GPU: {peak_mem:.2f} GB")

    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"Saved to {LORA_OUTPUT_DIR}")

    metadata = {
        "base_model": BASE_MODEL, "precision": PRECISION, "dataset": DATASET_NAME,
        "samples": len(dataset), "loss": stats.training_loss, "steps": stats.global_step,
        "runtime_s": stats.metrics.get("train_runtime"), "peak_gpu_gb": peak_mem,
        "lora_r": LORA_CONFIG["r"], "timestamp": datetime.now().isoformat(),
    }
    (Path(LORA_OUTPUT_DIR) / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    wandb.log({"final/loss": stats.training_loss, "final/peak_gpu_gb": peak_mem})
    print(f"\nW&B: {wandb.run.url}" if wandb.run else "")
    wandb.finish()


if __name__ == "__main__":
    main()
