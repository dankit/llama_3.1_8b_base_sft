"""Weights & Biases visualization for evaluation results.

Logs IFEval results to W&B for professional experiment tracking and comparison.
Run `wandb login` first to authenticate, or set WANDB_API_KEY environment variable.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import wandb

EVAL_RESULTS_DIR = "./eval_results"
PROJECT_NAME = "llama-3-1-8b-eval"


def load_results(results_dir: str = EVAL_RESULTS_DIR) -> list[dict]:
    """Load all JSON result files from directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"No results directory: {results_dir}")
        return []
    
    results = []
    for f in sorted(results_path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            data["_filepath"] = str(f)
            results.append(data)
            meta = data.get("metadata", {})
            print(f"  Found: {meta.get('run_name', f.stem)} ({meta.get('precision', '?')})")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Error loading {f}: {e}")
    
    return sorted(results, key=lambda x: x.get("metadata", {}).get("timestamp", ""))


def get_model_type(result: dict) -> str:
    """Return 'base' or 'finetuned' based on metadata."""
    meta = result.get("metadata", {})
    if meta.get("lora_path"):
        return "finetuned"
    run_name = meta.get("run_name", "").lower()
    if any(kw in run_name for kw in ["sft", "finetune", "lora", "trained"]):
        return "finetuned"
    return "base"


def log_to_wandb(results: list[dict], project: str = PROJECT_NAME):
    """Log evaluation results to Weights & Biases."""
    if not results:
        print("No results to log")
        return
    
    print(f"\nLogging {len(results)} runs to W&B project: {project}")
    
    for result in results:
        meta = result.get("metadata", {})
        config = result.get("config", {})
        timing = result.get("timing", {})
        training_meta = result.get("training_metadata", {})
        
        model_type = get_model_type(result)
        precision = meta.get("precision", "unknown")
        run_name = meta.get("run_name", "unknown")
        model_name = meta.get("model_name", "unknown")
        
        # Create descriptive run name
        wandb_run_name = f"{model_type}_{precision}_{run_name}"
        
        # Config captures all hyperparameters and settings
        wandb_config = {
            # Model info
            "model_name": model_name,
            "model_type": model_type,
            "precision": precision,
            "lora_path": meta.get("lora_path"),
            
            # Eval config
            "max_seq_length": config.get("max_seq_length"),
            "batch_size": config.get("batch_size"),
            "num_samples": config.get("num_samples"),
            "chat_template": config.get("chat_template"),
            "tasks": meta.get("tasks", []),
            
            # Generation config
            **(config.get("generation_config", {})),
            
            # Training info (if available)
            "training_dataset_samples": training_meta.get("dataset_samples"),
            "training_epochs": training_meta.get("epochs"),
            "training_learning_rate": training_meta.get("learning_rate"),
        }
        
        # Initialize run
        run = wandb.init(
            project=project,
            name=wandb_run_name,
            config=wandb_config,
            tags=[model_type, precision, "ifeval"],
            reinit=True,
        )
        
        # Log IFEval metrics
        ifeval_results = result.get("results", {}).get("ifeval", {})
        metrics = {
            "ifeval/prompt_strict_acc": ifeval_results.get("prompt_level_strict_acc", 0),
            "ifeval/prompt_loose_acc": ifeval_results.get("prompt_level_loose_acc", 0),
            "ifeval/inst_strict_acc": ifeval_results.get("inst_level_strict_acc", 0),
            "ifeval/inst_loose_acc": ifeval_results.get("inst_level_loose_acc", 0),
        }
        
        # Log timing metrics
        if timing:
            metrics["timing/generation_seconds"] = timing.get("generation_seconds", 0)
            metrics["timing/scoring_seconds"] = timing.get("scoring_seconds", 0)
            metrics["timing/total_seconds"] = timing.get("total_seconds", 0)
        
        # Log training metrics (if available)
        if training_meta:
            if training_meta.get("training_time_seconds"):
                metrics["training/time_seconds"] = training_meta["training_time_seconds"]
            if training_meta.get("training_loss"):
                metrics["training/final_loss"] = training_meta["training_loss"]
            if training_meta.get("peak_gpu_memory_gb"):
                metrics["training/peak_gpu_memory_gb"] = training_meta["peak_gpu_memory_gb"]
            if training_meta.get("training_samples_per_second"):
                metrics["training/samples_per_second"] = training_meta["training_samples_per_second"]
        
        # Log GPU memory from eval
        if meta.get("eval_gpu_memory_gb"):
            metrics["eval/gpu_memory_gb"] = meta["eval_gpu_memory_gb"]
        
        wandb.log(metrics)
        
        # Log summary table as artifact
        summary_table = wandb.Table(columns=["Metric", "Value"])
        summary_table.add_data("Model", model_name)
        summary_table.add_data("Type", model_type)
        summary_table.add_data("Precision", precision)
        summary_table.add_data("Prompt Strict Acc", f"{ifeval_results.get('prompt_level_strict_acc', 0):.4f}")
        summary_table.add_data("Prompt Loose Acc", f"{ifeval_results.get('prompt_level_loose_acc', 0):.4f}")
        summary_table.add_data("Inst Strict Acc", f"{ifeval_results.get('inst_level_strict_acc', 0):.4f}")
        summary_table.add_data("Inst Loose Acc", f"{ifeval_results.get('inst_level_loose_acc', 0):.4f}")
        
        wandb.log({"summary": summary_table})
        
        print(f"  Logged: {wandb_run_name}")
        run.finish()
    
    print(f"\nView results at: https://wandb.ai/{wandb.api.default_entity}/{project}")


def create_comparison_report(results: list[dict], project: str = PROJECT_NAME):
    """Create a W&B comparison report if we have base and finetuned results."""
    base_results = [r for r in results if get_model_type(r) == "base"]
    ft_results = [r for r in results if get_model_type(r) == "finetuned"]
    
    if not base_results or not ft_results:
        print("\nNeed both base and finetuned results for comparison report")
        return
    
    # Use latest of each
    base = base_results[-1]
    ft = ft_results[-1]
    
    base_ifeval = base.get("results", {}).get("ifeval", {})
    ft_ifeval = ft.get("results", {}).get("ifeval", {})
    
    # Create comparison table
    run = wandb.init(
        project=project,
        name="comparison_base_vs_finetuned",
        tags=["comparison"],
        reinit=True,
    )
    
    comparison_table = wandb.Table(columns=["Metric", "Base", "Finetuned", "Improvement"])
    
    metrics_to_compare = [
        ("Prompt Strict Acc", "prompt_level_strict_acc"),
        ("Prompt Loose Acc", "prompt_level_loose_acc"),
        ("Inst Strict Acc", "inst_level_strict_acc"),
        ("Inst Loose Acc", "inst_level_loose_acc"),
    ]
    
    for display_name, key in metrics_to_compare:
        base_val = base_ifeval.get(key, 0)
        ft_val = ft_ifeval.get(key, 0)
        improvement = ft_val - base_val
        comparison_table.add_data(
            display_name,
            f"{base_val:.4f}",
            f"{ft_val:.4f}",
            f"{improvement:+.4f} ({improvement/base_val*100:+.1f}%)" if base_val > 0 else f"{improvement:+.4f}"
        )
    
    wandb.log({"base_vs_finetuned": comparison_table})
    
    # Log bar chart data
    wandb.log({
        "comparison/prompt_strict_acc": wandb.plot.bar(
            wandb.Table(data=[["Base", base_ifeval.get("prompt_level_strict_acc", 0)],
                              ["Finetuned", ft_ifeval.get("prompt_level_strict_acc", 0)]],
                        columns=["Model", "Accuracy"]),
            "Model", "Accuracy", title="Prompt-Level Strict Accuracy"
        ),
        "comparison/inst_strict_acc": wandb.plot.bar(
            wandb.Table(data=[["Base", base_ifeval.get("inst_level_strict_acc", 0)],
                              ["Finetuned", ft_ifeval.get("inst_level_strict_acc", 0)]],
                        columns=["Model", "Accuracy"]),
            "Model", "Accuracy", title="Instruction-Level Strict Accuracy"
        ),
    })
    
    print(f"\nComparison logged!")
    run.finish()


def log_mmlu_to_wandb(results_dir: str = EVAL_RESULTS_DIR, project: str = PROJECT_NAME):
    """Log MMLU/tinyMMLU results to W&B (different format from IFEval)."""
    results_path = Path(results_dir)
    mmlu_files = list(results_path.glob("mmlu*.json"))
    
    if not mmlu_files:
        print("No MMLU result files found")
        return
    
    print(f"Logging {len(mmlu_files)} MMLU runs to W&B project: {project}")
    
    for f in mmlu_files:
        data = json.loads(f.read_text())
        
        # Extract model info
        model_name = data.get("model_name", "unknown")
        cfg = data.get("config", {})
        mmlu_cfg = data.get("configs", {}).get("tinyMMLU", {}).get("metadata", {})
        
        # Get accuracy from results
        mmlu_results = data.get("results", {}).get("tinyMMLU", {})
        acc = mmlu_results.get("acc_norm,none", 0)
        
        # Derive run name from filename
        run_name = f.stem.replace("_results", "")
        
        run = wandb.init(
            project=project,
            name=run_name,
            config={
                "model_name": model_name,
                "dtype": mmlu_cfg.get("dtype", cfg.get("model_dtype")),
                "batch_size": cfg.get("batch_size"),
                "n_samples": data.get("n-samples", {}).get("tinyMMLU", {}).get("effective"),
            },
            tags=["mmlu", mmlu_cfg.get("dtype", "unknown")],
            reinit=True,
        )
        
        wandb.log({
            "mmlu/acc_norm": acc,
            "eval_time_seconds": float(data.get("total_evaluation_time_seconds", 0)),
        })
        
        print(f"Logged: {run_name} (acc={acc:.4f})")
        run.finish()
    
    print(f"\nView at: https://wandb.ai/{wandb.api.default_entity}/{project}")


def main():
    parser = argparse.ArgumentParser(description="Log evaluation results to Weights & Biases")
    parser.add_argument("--results-dir", default=EVAL_RESULTS_DIR, help="Directory with result JSONs")
    parser.add_argument("--project", default=PROJECT_NAME, help="W&B project name")
    parser.add_argument("--compare", action="store_true", help="Also create comparison report")
    parser.add_argument("--mmlu", action="store_true", help="Log MMLU results instead of IFEval (after running lm_eval)")
    args = parser.parse_args()
    
    if args.mmlu:
        log_mmlu_to_wandb(args.results_dir, args.project)
        return
    
    print("Loading evaluation results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    log_to_wandb(results, args.project)
    
    if args.compare:
        create_comparison_report(results, args.project)
    
    print("\nDone! View your runs in the W&B dashboard.")


if __name__ == "__main__":
    main()
