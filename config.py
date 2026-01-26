"""Shared configuration for train/eval/inference scripts."""

import torch

# =============================================================================
# MODEL
# =============================================================================
# For SFT experiments: train on BASE, compare against both BASE and INSTRUCT
BASE_MODEL = "unsloth/Llama-3.1-8B"  # Base model for training
INSTRUCT_MODEL = "unsloth/Llama-3.1-8B-Instruct"  # Reference for comparison
MAX_SEQ_LENGTH = 2048
CHAT_TEMPLATE = "llama-3.1"
USE_CHAT_TEMPLATE = True

# =============================================================================
# PRECISION: "4bit", "8bit", "bf16", "fp16"
# =============================================================================
PRECISION = "bf16"  # A100 40GB can handle full bf16

PRECISION_MAP = {
    "4bit": {"load_in_4bit": True, "load_in_8bit": False, "dtype": None},
    "8bit": {"load_in_4bit": False, "load_in_8bit": True, "dtype": None},
    "bf16": {"load_in_4bit": False, "load_in_8bit": False, "dtype": torch.bfloat16},
    "fp16": {"load_in_4bit": False, "load_in_8bit": False, "dtype": torch.float16},
}

def get_precision_kwargs(precision: str = PRECISION) -> dict:
    """Get model loading kwargs for given precision."""
    if precision not in PRECISION_MAP:
        raise ValueError(f"Invalid precision: {precision}. Use: {list(PRECISION_MAP.keys())}")
    return PRECISION_MAP[precision]

# =============================================================================
# PATHS
# =============================================================================
TRAINING_OUTPUT_DIR = "./outputs"
LORA_OUTPUT_DIR = f"./lora_model_{PRECISION}"
EVAL_RESULTS_DIR = "./eval_results"

# =============================================================================
# TRAINING
# =============================================================================
LORA_CONFIG = {
    "r": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": False,
    "random_state": 42,
    "use_rslora": False,
    "loftq_config": None,
}

# =============================================================================
# DATASET
# =============================================================================
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train"
DATASET_NUM_SAMPLES = None  # Use full 52k dataset

# =============================================================================
# GENERATION (inference)
# =============================================================================
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "use_cache": True,
}

# =============================================================================
# EVALUATION
# =============================================================================
EVAL_TASKS = ["ifeval", "tinyMMLU"]
#"auto" for auto detect, else set to int
EVAL_BATCH_SIZE = 64