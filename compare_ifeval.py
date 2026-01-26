"""Compare two models' IFEval performance by instruction type.

Usage:
    python compare_ifeval.py --model1 eval_responses/model_a.json --model2 eval_responses/model_b.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Import IFEval instructions
import sys
local_ifeval = Path(__file__).parent / "instruction_following_eval"
if local_ifeval.exists():
    sys.path.insert(0, str(Path(__file__).parent))

from instruction_following_eval import instructions_registry


def check_instruction_following(prompt: str, response: str, instruction_id_list: list, kwargs: list) -> dict:
    """Check if response follows all instructions. Returns per-instruction results."""
    results = {"strict": [], "loose": [], "instruction_ids": instruction_id_list}
    
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
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(inst_id)
        if instruction_cls is None:
            print(f"Warning: Unknown instruction {inst_id}, skipping")
            results["strict"].append(False)
            results["loose"].append(False)
            continue
            
        instruction = instruction_cls(inst_id)
        valid_keys = instruction.get_instruction_args_keys()
        filtered_kwargs = {k: v for k, v in inst_kwargs.items() if k in valid_keys}
        instruction.build_description(**filtered_kwargs)
        
        strict_pass = instruction.check_following(response)
        loose_pass = strict_pass
        if hasattr(instruction, "check_following_loose"):
            loose_pass = instruction.check_following_loose(response)
        
        results["strict"].append(strict_pass)
        results["loose"].append(loose_pass)
    
    return results


def score_by_instruction_type(samples: list) -> dict:
    """Score samples and group results by instruction type."""
    # Track pass/total for each instruction type
    by_type = defaultdict(lambda: {"strict_pass": 0, "loose_pass": 0, "total": 0})
    # Also track by category (the prefix before the colon)
    by_category = defaultdict(lambda: {"strict_pass": 0, "loose_pass": 0, "total": 0})
    
    for sample in samples:
        result = check_instruction_following(
            prompt=sample["prompt"],
            response=sample["response"],
            instruction_id_list=sample["instruction_id_list"],
            kwargs=sample["kwargs"],
        )
        
        for i, inst_id in enumerate(result["instruction_ids"]):
            by_type[inst_id]["total"] += 1
            by_type[inst_id]["strict_pass"] += int(result["strict"][i])
            by_type[inst_id]["loose_pass"] += int(result["loose"][i])
            
            # Extract category (e.g., "punctuation" from "punctuation:no_comma")
            category = inst_id.split(":")[0] if ":" in inst_id else inst_id
            by_category[category]["total"] += 1
            by_category[category]["strict_pass"] += int(result["strict"][i])
            by_category[category]["loose_pass"] += int(result["loose"][i])
    
    return dict(by_type), dict(by_category)


def load_responses(path: str) -> tuple:
    """Load response file and return samples and model name."""
    with open(path) as f:
        data = json.load(f)
    
    model_name = data["config"]["model_name"]
    if data["config"].get("lora_path"):
        model_name = Path(data["config"]["lora_path"]).name
    
    return data["samples"], model_name


def compare_models(model1_path: str, model2_path: str, use_strict: bool = True):
    """Compare two models' IFEval performance by instruction type."""
    
    print("Loading model responses...")
    samples1, name1 = load_responses(model1_path)
    samples2, name2 = load_responses(model2_path)
    
    print(f"Model 1: {name1} ({len(samples1)} samples)")
    print(f"Model 2: {name2} ({len(samples2)} samples)")
    
    print("\nScoring model 1...")
    by_type1, by_cat1 = score_by_instruction_type(samples1)
    
    print("Scoring model 2...")
    by_type2, by_cat2 = score_by_instruction_type(samples2)
    
    metric = "strict_pass" if use_strict else "loose_pass"
    metric_label = "strict" if use_strict else "loose"
    
    # Print category-level comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON BY CATEGORY ({metric_label})")
    print(f"{'='*80}")
    
    all_categories = sorted(set(by_cat1.keys()) | set(by_cat2.keys()))
    
    # Header
    print(f"{'Category':<25} {'Model 1':>12} {'Model 2':>12} {'Diff':>10} {'Winner':<20}")
    print("-" * 80)
    
    category_wins = {name1: 0, name2: 0, "tie": 0}
    
    for cat in all_categories:
        stats1 = by_cat1.get(cat, {"strict_pass": 0, "loose_pass": 0, "total": 0})
        stats2 = by_cat2.get(cat, {"strict_pass": 0, "loose_pass": 0, "total": 0})
        
        acc1 = stats1[metric] / stats1["total"] if stats1["total"] > 0 else 0
        acc2 = stats2[metric] / stats2["total"] if stats2["total"] > 0 else 0
        diff = acc1 - acc2
        
        if abs(diff) < 0.001:
            winner = "tie"
            category_wins["tie"] += 1
        elif diff > 0:
            winner = f"<- M1 wins"
            category_wins[name1] += 1
        else:
            winner = f"-> M2 wins"
            category_wins[name2] += 1
        
        print(f"{cat:<25} {acc1:>11.1%} {acc2:>11.1%} {diff:>+9.1%} {winner:<15}")
    
    print("-" * 80)
    print(f"Category wins: {name1}: {category_wins[name1]}, {name2}: {category_wins[name2]}, ties: {category_wins['tie']}")
    
    # Print detailed instruction-level comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON BY INSTRUCTION TYPE ({metric_label})")
    print(f"{'='*80}")
    
    all_types = sorted(set(by_type1.keys()) | set(by_type2.keys()))
    
    print(f"{'Instruction Type':<45} {'Model 1':>10} {'Model 2':>10} {'Diff':>8} {'Winner'}")
    print("-" * 90)
    
    type_wins = {name1: 0, name2: 0, "tie": 0}
    comparison_data = []
    
    for inst_type in all_types:
        stats1 = by_type1.get(inst_type, {"strict_pass": 0, "loose_pass": 0, "total": 0})
        stats2 = by_type2.get(inst_type, {"strict_pass": 0, "loose_pass": 0, "total": 0})
        
        acc1 = stats1[metric] / stats1["total"] if stats1["total"] > 0 else 0
        acc2 = stats2[metric] / stats2["total"] if stats2["total"] > 0 else 0
        diff = acc1 - acc2
        
        if abs(diff) < 0.001:
            winner = "tie"
            type_wins["tie"] += 1
        elif diff > 0:
            winner = "<- M1"
            type_wins[name1] += 1
        else:
            winner = "-> M2"
            type_wins[name2] += 1
        
        comparison_data.append({
            "instruction_type": inst_type,
            "model1_acc": acc1,
            "model2_acc": acc2,
            "model1_count": f"{stats1[metric]}/{stats1['total']}",
            "model2_count": f"{stats2[metric]}/{stats2['total']}",
            "diff": diff,
            "winner": winner
        })
        
        print(f"{inst_type:<45} {acc1:>9.1%} {acc2:>9.1%} {diff:>+7.1%} {winner}")
    
    print("-" * 90)
    print(f"Instruction type wins: {name1}: {type_wins[name1]}, {name2}: {type_wins[name2]}, ties: {type_wins['tie']}")
    
    # Overall accuracy
    total1_pass = sum(s["strict_pass"] if use_strict else s["loose_pass"] for s in by_type1.values())
    total1_count = sum(s["total"] for s in by_type1.values())
    total2_pass = sum(s["strict_pass"] if use_strict else s["loose_pass"] for s in by_type2.values())
    total2_count = sum(s["total"] for s in by_type2.values())
    
    overall1 = total1_pass / total1_count if total1_count > 0 else 0
    overall2 = total2_pass / total2_count if total2_count > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"OVERALL INSTRUCTION-LEVEL ACCURACY ({metric_label})")
    print(f"{'='*80}")
    print(f"{name1}: {overall1:.1%} ({total1_pass}/{total1_count})")
    print(f"{name2}: {overall2:.1%} ({total2_pass}/{total2_count})")
    print(f"Difference: {overall1 - overall2:+.1%}")
    
    # Return data for further analysis
    return {
        "model1": {"name": name1, "by_type": by_type1, "by_category": by_cat1, "overall": overall1},
        "model2": {"name": name2, "by_type": by_type2, "by_category": by_cat2, "overall": overall2},
        "comparison": comparison_data
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two models' IFEval performance by instruction type")
    parser.add_argument("--model1", required=True, help="Path to first model's response JSON")
    parser.add_argument("--model2", required=True, help="Path to second model's response JSON")
    parser.add_argument("--loose", action="store_true", help="Use loose accuracy instead of strict")
    parser.add_argument("--output", help="Optional: save comparison results to JSON")
    args = parser.parse_args()
    
    results = compare_models(args.model1, args.model2, use_strict=not args.loose)
    
    if args.output:
        # Convert for JSON serialization
        output_data = {
            "model1": {
                "name": results["model1"]["name"],
                "by_type": {k: dict(v) for k, v in results["model1"]["by_type"].items()},
                "by_category": {k: dict(v) for k, v in results["model1"]["by_category"].items()},
                "overall": results["model1"]["overall"]
            },
            "model2": {
                "name": results["model2"]["name"],
                "by_type": {k: dict(v) for k, v in results["model2"]["by_type"].items()},
                "by_category": {k: dict(v) for k, v in results["model2"]["by_category"].items()},
                "overall": results["model2"]["overall"]
            },
            "comparison": results["comparison"]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved comparison to: {args.output}")


if __name__ == "__main__":
    main()
