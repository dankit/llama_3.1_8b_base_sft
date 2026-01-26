"""Score saved responses using IFEval checks (CPU step).

This script loads responses from eval_generate.py and runs the
IFEval instruction-following checks. No GPU required.

Uses lm-evaluation-harness's built-in IFEval implementation.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Import IFEval instructions - try local folder first, then lm-eval
import sys
from pathlib import Path

# Add local instruction_following_eval to path if it exists
local_ifeval = Path(__file__).parent / "instruction_following_eval"
if local_ifeval.exists():
    sys.path.insert(0, str(Path(__file__).parent))

try:
    from instruction_following_eval import instructions_registry
except ImportError:
    raise ImportError(
        "Could not import instruction_following_eval.\n"
        "Run the setup script first:\n"
        "  powershell -ExecutionPolicy Bypass -File setup_ifeval.ps1\n"
        "Then verify all dependencies are installed (e.g. langdetect). Or download manually from:\n"
        " https://github.com/google-research/google-research/tree/master/instruction_following_eval"
    )


def check_instruction_following(prompt: str, response: str, instruction_id_list: list, kwargs: list) -> dict:
    """Check if response follows all instructions.
    
    Returns dict with strict and loose pass/fail for each instruction.
    """
    results = {"strict": [], "loose": []}
    
    # Parse kwargs (stored as JSON strings in the dataset)
    parsed_kwargs = []
    for kw in kwargs:
        if isinstance(kw, str):
            try:
                parsed_kwargs.append(json.loads(kw) if kw else {})
            except json.JSONDecodeError:
                parsed_kwargs.append({})
        else:
            parsed_kwargs.append(kw if kw else {})
    
    for inst_id, inst_kwargs in zip(instruction_id_list, parsed_kwargs):
        # Get the instruction checker from lm-eval's registry
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(inst_id)
        if instruction_cls is None:
            print(f"Warning: Unknown instruction {inst_id}, skipping")
            results["strict"].append(False)
            results["loose"].append(False)
            continue
            
        instruction = instruction_cls(inst_id)
        
        # Filter kwargs to only include keys this instruction accepts
        valid_keys = instruction.get_instruction_args_keys()
        filtered_kwargs = {k: v for k, v in inst_kwargs.items() if k in valid_keys}
        instruction.build_description(**filtered_kwargs)
        
        # Check strict (exact) and loose (flexible) following
        strict_pass = instruction.check_following(response)
        loose_pass = strict_pass  # Default to strict
        
        # Some instructions have a separate loose check
        if hasattr(instruction, "check_following_loose"):
            loose_pass = instruction.check_following_loose(response)
        
        results["strict"].append(strict_pass)
        results["loose"].append(loose_pass)
    
    return results


def score_responses(samples: list) -> dict:
    """Score all responses and compute metrics."""
    
    # Per-instruction accuracy
    strict_correct = 0
    strict_total = 0
    loose_correct = 0
    loose_total = 0
    
    # Per-prompt accuracy (all instructions must pass)
    prompt_strict_correct = 0
    prompt_loose_correct = 0
    prompt_total = len(samples)
    
    scored_samples = []
    
    for sample in samples:
        result = check_instruction_following(
            prompt=sample["prompt"],
            response=sample["response"],
            instruction_id_list=sample["instruction_id_list"],
            kwargs=sample["kwargs"],
        )
        
        # Instruction-level stats
        strict_correct += sum(result["strict"])
        strict_total += len(result["strict"])
        loose_correct += sum(result["loose"])
        loose_total += len(result["loose"])
        
        # Prompt-level stats (all instructions must pass)
        if all(result["strict"]):
            prompt_strict_correct += 1
        if all(result["loose"]):
            prompt_loose_correct += 1
        
        scored_samples.append({
            **sample,
            "scores": result,
            "prompt_strict_pass": all(result["strict"]),
            "prompt_loose_pass": all(result["loose"]),
        })
    
    metrics = {
        # Instruction-level accuracy (like lm-eval's inst_level metrics)
        "ifeval/inst_level_strict_acc": strict_correct / strict_total if strict_total > 0 else 0,
        "ifeval/inst_level_loose_acc": loose_correct / loose_total if loose_total > 0 else 0,
        # Prompt-level accuracy (like lm-eval's prompt_level metrics)
        "ifeval/prompt_level_strict_acc": prompt_strict_correct / prompt_total if prompt_total > 0 else 0,
        "ifeval/prompt_level_loose_acc": prompt_loose_correct / prompt_total if prompt_total > 0 else 0,
    }
    
    return metrics, scored_samples


def main():
    parser = argparse.ArgumentParser(description="Score saved responses with IFEval checks")
    parser.add_argument("--input", required=True, help="Path to responses JSON from eval_generate.py")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--save-samples", action="store_true", help="Include scored samples in output")
    args = parser.parse_args()
    
    # Load responses
    print(f"Loading responses from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    
    config = data["config"]
    gen_metadata = data["metadata"]
    samples = data["samples"]
    print(f"Loaded {len(samples)} samples")
    
    # Score
    print("Running IFEval checks...")
    start_time = time.time()
    metrics, scored_samples = score_responses(samples)
    scoring_time = time.time() - start_time
    
    # Build output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add timing metrics
    metrics["timing/generation_seconds"] = gen_metadata["generation_time_seconds"]
    metrics["timing/scoring_seconds"] = scoring_time
    metrics["timing/total_seconds"] = gen_metadata["generation_time_seconds"] + scoring_time
    
    if gen_metadata.get("gpu_memory_gb"):
        metrics["eval/gpu_memory_gb"] = gen_metadata["gpu_memory_gb"]
    
    output_data = {
        "metadata": {
            "model_name": config["model_name"],
            "lora_path": config["lora_path"],
            "precision": config["precision"],
            "run_name": gen_metadata["run_name"],
            "timestamp": timestamp,
            "tasks": ["ifeval"],
            "eval_gpu_memory_gb": gen_metadata.get("gpu_memory_gb"),
        },
        "config": {
            "max_seq_length": config["max_seq_length"],
            "batch_size": config["batch_size"],
            "generation_config": config["generation_config"],
            "chat_template": config["chat_template"],
            "num_samples": gen_metadata["num_samples"],
        },
        "timing": {
            "generation_seconds": gen_metadata["generation_time_seconds"],
            "scoring_seconds": scoring_time,
            "total_seconds": gen_metadata["generation_time_seconds"] + scoring_time,
        },
        "results": {
            "ifeval": {
                "inst_level_strict_acc": metrics["ifeval/inst_level_strict_acc"],
                "inst_level_loose_acc": metrics["ifeval/inst_level_loose_acc"],
                "prompt_level_strict_acc": metrics["ifeval/prompt_level_strict_acc"],
                "prompt_level_loose_acc": metrics["ifeval/prompt_level_loose_acc"],
            }
        },
        "metrics_flat": metrics,
    }
    
    if args.save_samples:
        output_data["scored_samples"] = scored_samples
    
    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_name = gen_metadata["run_name"]
    precision = config["precision"]
    output_path = Path(args.output_dir) / f"{run_name}_{precision}_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print results
    print(f"\n{'='*50}")
    print("IFEVAL RESULTS")
    print(f"{'='*50}")
    print(f"Model: {config['model_name']}")
    print(f"Precision: {config['precision']}")
    print(f"Max Seq Length: {config['max_seq_length']}")
    print(f"Samples: {gen_metadata['num_samples']}")
    print()
    print(f"Prompt-level strict accuracy: {metrics['ifeval/prompt_level_strict_acc']:.4f}")
    print(f"Prompt-level loose accuracy:  {metrics['ifeval/prompt_level_loose_acc']:.4f}")
    print(f"Inst-level strict accuracy:   {metrics['ifeval/inst_level_strict_acc']:.4f}")
    print(f"Inst-level loose accuracy:    {metrics['ifeval/inst_level_loose_acc']:.4f}")
    print()
    print(f"Generation time: {gen_metadata['generation_time_seconds']:.1f}s")
    print(f"Scoring time: {scoring_time:.1f}s")
    print(f"Total time: {gen_metadata['generation_time_seconds'] + scoring_time:.1f}s")
    print()
    print(f"Saved: {output_path}")
    print(f"\nLog to W&B: python eval_wandb.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
