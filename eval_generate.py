"""Generate LLM responses for evaluation (GPU step).

This script generates responses and saves them to disk.
Scoring happens separately in eval_score.py (CPU only).

Supports both Unsloth (default) and standard HuggingFace loading (--use-hf).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from config import (
    BASE_MODEL, MAX_SEQ_LENGTH, PRECISION, get_precision_kwargs,
    LORA_OUTPUT_DIR, EVAL_BATCH_SIZE, CHAT_TEMPLATE, USE_CHAT_TEMPLATE,
)

# Deterministic generation for reproducible evaluation
EVAL_GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "do_sample": False,  # Greedy decoding
}


def setup_tokenizer(tokenizer):
    """Configure tokenizer for batched generation."""
    # Set pad token - Llama 3.1 has a dedicated pad token <|finetune_right_pad_id|> (128004)
    # This avoids confusion with EOS token during generation
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        # Try to use Llama 3.1's dedicated pad token
        LLAMA_31_PAD_TOKEN_ID = 128004  # <|finetune_right_pad_id|>
        if LLAMA_31_PAD_TOKEN_ID < len(tokenizer):
            tokenizer.pad_token_id = LLAMA_31_PAD_TOKEN_ID
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(LLAMA_31_PAD_TOKEN_ID)
            print(f"Using Llama 3.1 pad token: {tokenizer.pad_token!r} (id={LLAMA_31_PAD_TOKEN_ID})")
        else:
            # Fallback for other models
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Warning: Using eos_token as pad_token: {tokenizer.pad_token}")
    
    print(f"Tokenizer: padding_side={tokenizer.padding_side}, pad_token={tokenizer.pad_token!r}")
    return tokenizer


def load_model_unsloth(model_name: str, lora_path: str = None, precision: str = PRECISION, use_chat_template: bool = True):
    """Load model using Unsloth (faster, optimized)."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    
    prec = get_precision_kwargs(precision)
    load_path = lora_path if lora_path and Path(lora_path).exists() else model_name
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"Loading {load_path} ({precision}) via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=prec["dtype"],
        load_in_4bit=prec["load_in_4bit"],
        load_in_8bit=prec["load_in_8bit"],
    )
    
    if use_chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    
    tokenizer = setup_tokenizer(tokenizer)
    FastLanguageModel.for_inference(model)
    
    if torch.cuda.is_available():
        load_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Model loaded, GPU memory: {load_mem:.2f} GB")
    
    return model, tokenizer


def load_model_hf(model_name: str, lora_path: str = None, precision: str = PRECISION, use_chat_template: bool = True):
    """Load model using standard HuggingFace (for comparison/debugging)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    load_path = lora_path if lora_path and Path(lora_path).exists() else model_name
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"Loading {load_path} ({precision}) via HuggingFace...")
    
    # Configure quantization
    quantization_config = None
    torch_dtype = None
    
    if precision == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif precision == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.float16
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = setup_tokenizer(tokenizer)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="sdpa",  # Use PyTorch SDPA (similar to Flash Attention)
    )
    
    # Load LoRA if specified
    if lora_path and Path(lora_path).exists() and lora_path != model_name:
        print(f"Loading LoRA adapters from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    
    if torch.cuda.is_available():
        load_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Model loaded, GPU memory: {load_mem:.2f} GB")
    
    return model, tokenizer


def load_model(model_name: str, lora_path: str = None, precision: str = PRECISION, use_chat_template: bool = True, use_hf: bool = False):
    """Load model using either Unsloth or HuggingFace."""
    if use_hf:
        return load_model_hf(model_name, lora_path, precision, use_chat_template)
    else:
        return load_model_unsloth(model_name, lora_path, precision, use_chat_template)


def generate_responses(model, tokenizer, prompts: list[str], batch_size: int = 1, use_chat_template: bool = True) -> list[str]:
    """Generate responses for a list of prompts."""
    responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]
        
        if use_chat_template:
            # Format as chat messages
            batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]
            batch_texts = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_messages
            ]
        else:
            # Raw prompts for base models
            batch_texts = batch_prompts
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Track actual (non-padded) input lengths per sample using attention mask
        # With left padding, actual tokens are at the END, so sum of attention_mask gives true length
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EVAL_GENERATION_CONFIG["max_new_tokens"],
                do_sample=EVAL_GENERATION_CONFIG["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # The padded input length (where generation starts in the output tensor)
        padded_input_len = inputs["input_ids"].shape[1]
        
        for j, output in enumerate(outputs):
            # With left padding: output is [PAD...PAD, PROMPT, GENERATED]
            # Generation starts at padded_input_len for all samples
            # We slice from there to get only the newly generated tokens
            generated_tokens = output[padded_input_len:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
    
    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate responses for evaluation")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--lora", default=None, nargs="?", const=LORA_OUTPUT_DIR)
    parser.add_argument("--precision", default=PRECISION, choices=["4bit", "8bit", "bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--output-dir", default="./eval_responses")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--no-chat-template", action="store_true", help="Don't apply chat template (for base models)")
    parser.add_argument("--use-hf", action="store_true", help="Use HuggingFace instead of Unsloth (for debugging)")
    args = parser.parse_args()
    
    use_chat = USE_CHAT_TEMPLATE and not args.no_chat_template
    
    # Load model
    model, tokenizer = load_model(args.model, args.lora, args.precision, use_chat, use_hf=args.use_hf)
    
    # Load IFEval dataset
    print("Loading IFEval dataset...")
    dataset = load_dataset("google/IFEval", split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Loaded {len(dataset)} samples")
    
    # Generate responses
    print("Generating responses...")
    start_time = time.time()
    
    prompts = dataset["prompt"]
    responses = generate_responses(model, tokenizer, prompts, args.batch_size, use_chat)
    
    generation_time = time.time() - start_time
    
    # Get GPU memory stats
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    # Build output data
    run_name = Path(args.lora).name if args.lora else args.model.replace("/", "_")
    if args.use_hf:
        run_name += "_hf"  # Mark HuggingFace runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        "config": {
            "model_name": args.model,
            "lora_path": args.lora,
            "precision": args.precision,
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": args.batch_size,
            "generation_config": EVAL_GENERATION_CONFIG,
            "chat_template": CHAT_TEMPLATE if use_chat else None,
            "loader": "huggingface" if args.use_hf else "unsloth",
        },
        "metadata": {
            "run_name": run_name,
            "timestamp": timestamp,
            "num_samples": len(dataset),
            "generation_time_seconds": generation_time,
            "gpu_memory_gb": gpu_memory,
        },
        "samples": [
            {
                "key": dataset[i]["key"],
                "prompt": dataset[i]["prompt"],
                "response": responses[i],
                "instruction_id_list": dataset[i]["instruction_id_list"],
                "kwargs": dataset[i]["kwargs"],
            }
            for i in range(len(dataset))
        ],
    }
    
    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{run_name}_{args.precision}_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"  Samples: {len(dataset)}")
    print(f"  Time: {generation_time:.1f}s ({generation_time/60:.2f} min)")
    print(f"  GPU Memory: {gpu_memory:.2f} GB" if gpu_memory else "  GPU Memory: N/A")
    print(f"  Saved: {output_path}")
    print(f"\nNext step: python eval_score.py --input {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
