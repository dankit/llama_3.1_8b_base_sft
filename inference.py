"""Interactive inference with Unsloth models."""

import argparse
import time
from pathlib import Path
import torch

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from config import (
    BASE_MODEL, MAX_SEQ_LENGTH, PRECISION, get_precision_kwargs,
    LORA_OUTPUT_DIR, GENERATION_CONFIG, USE_CHAT_TEMPLATE, CHAT_TEMPLATE,
)


def load_model(lora_path: str = None, use_chat_template: bool = USE_CHAT_TEMPLATE):
    """Load model for inference, optionally with LoRA adapters."""
    prec = get_precision_kwargs()
    load_path = lora_path if lora_path and Path(lora_path).exists() else BASE_MODEL
    
    print(f"Loading {load_path} ({PRECISION})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path, max_seq_length=MAX_SEQ_LENGTH,
        dtype=prec["dtype"], load_in_4bit=prec["load_in_4bit"], load_in_8bit=prec["load_in_8bit"],
    )
    
    if use_chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, system_prompt: str = None, 
             use_chat_template: bool = USE_CHAT_TEMPLATE, return_metrics: bool = False, **kwargs):
    """Generate a response for a single prompt."""
    gen_config = {**GENERATION_CONFIG, **kwargs}
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    if use_chat_template:
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    else:
        inputs = tokenizer("\n\n".join(m["content"] for m in messages), return_tensors="pt").input_ids.to("cuda")
    
    input_len = inputs.shape[1]
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = model.generate(input_ids=inputs, **gen_config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    if return_metrics:
        tokens = outputs.shape[1] - input_len
        return response, {"tokens": tokens, "time": round(elapsed, 2), "tok/s": round(tokens/elapsed, 1)}
    return response


def interactive(model, tokenizer, system_prompt: str = None, use_chat_template: bool = USE_CHAT_TEMPLATE, **kwargs):
    """Interactive chat loop."""
    gen_config = {**GENERATION_CONFIG, **kwargs}
    
    print("\n" + "=" * 50)
    print("CHAT MODE (type 'quit' to exit, 'clear' to reset)")
    print("=" * 50)
    
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": system_prompt}] if system_prompt else []
            print("History cleared.")
            continue
        
        history.append({"role": "user", "content": user_input})
        
        if use_chat_template:
            inputs = tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        else:
            inputs = tokenizer("\n\n".join(m["content"] for m in history), return_tensors="pt").input_ids.to("cuda")
        
        # Truncate if too long
        if inputs.shape[1] > MAX_SEQ_LENGTH - gen_config["max_new_tokens"]:
            history = ([history[0]] if system_prompt else []) + history[-4:]
            inputs = tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        input_len = inputs.shape[1]
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model.generate(input_ids=inputs, **gen_config)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        tokens = outputs.shape[1] - input_len
        
        history.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}")
        print(f"  [{tokens} tok, {elapsed:.2f}s, {tokens/elapsed:.1f} tok/s]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=None, nargs="?", const=LORA_OUTPUT_DIR)
    parser.add_argument("--system", default="You are a helpful AI assistant.")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    args = parser.parse_args()
    
    use_chat = USE_CHAT_TEMPLATE and not args.no_chat_template
    model, tokenizer = load_model(args.lora, use_chat)
    
    # Quick test
    print("\n" + "=" * 50)
    response, metrics = generate(model, tokenizer, "Explain machine learning briefly.", 
                                  use_chat_template=use_chat, return_metrics=True)
    print(f"Test: {response[:200]}...")
    print(f"[{metrics['tokens']} tok, {metrics['time']}s, {metrics['tok/s']} tok/s]")
    
    if not args.no_interactive:
        interactive(model, tokenizer, args.system, use_chat)


if __name__ == "__main__":
    main()
