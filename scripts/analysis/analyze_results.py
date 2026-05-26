#!/usr/bin/env python
"""Analyze FedLEASE experiment results across multiple seeds and compare with paper."""

import os
import json
import argparse
import numpy as np
from collections import defaultdict


# Paper Table 1 results (FedLEASE row)
PAPER_RESULTS = {
    "sst2": {"mean": 93.33, "std": 0.30},
    "qnli": {"mean": 87.22, "std": 1.16},
    "mrpc": {"mean": 86.93, "std": 0.68},
    "qqp":  {"mean": 83.57, "std": 0.96},
    "average": {"mean": 87.76, "std": 0.78},
}

TASK_ORDER = ["sst2", "qnli", "mrpc", "qqp", "average"]


def load_results(output_dir, seeds, model_name="roberta-large", num_clients=16):
    """Load training_history.json from each seed directory."""
    results = {}
    for seed in seeds:
        dir_name = f"{model_name.replace('/', '_')}_multi_task_federated_{num_clients}_seed{seed}"
        history_path = os.path.join(output_dir, dir_name, "proposed_m2", "training_history.json")

        if not os.path.exists(history_path):
            print(f"  [MISSING] Seed {seed}: {history_path}")
            continue

        with open(history_path) as f:
            data = json.load(f)

        # Extract final task metrics if available
        if "final_task_metrics" in data and data["final_task_metrics"]:
            results[seed] = data["final_task_metrics"]
            print(f"  [OK] Seed {seed}: {data['final_task_metrics']}")
        else:
            # Fall back to computing from client_scores
            client_scores = data.get("client_scores", {})
            task_accs = defaultdict(list)

            for client_id, scores_list in client_scores.items():
                if scores_list:
                    final_acc = scores_list[-1].get("eval_accuracy", 0) * 100
                    # Need task_info to map client_id to task, try to infer from round summaries
                    task_accs["unknown"].append(final_acc)

            if task_accs:
                results[seed] = {k: sum(v)/len(v) for k, v in task_accs.items()}
                print(f"  [OK] Seed {seed} (computed): {results[seed]}")

    return results


def compute_statistics(results):
    """Compute mean ± std per task across seeds."""
    task_values = defaultdict(list)
    for seed, metrics in results.items():
        for task, value in metrics.items():
            task_values[task].append(value)

    stats = {}
    for task in TASK_ORDER:
        if task in task_values and task_values[task]:
            values = np.array(task_values[task])
            stats[task] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values.tolist(),
                "n": len(values),
            }
    return stats


def print_comparison_table(stats):
    """Print comparison table between our results and paper results."""
    print("\n" + "=" * 90)
    print("FedLEASE Reproduction Results vs. Paper (Table 1)")
    print("=" * 90)

    header = f"{'Task':<12} {'Paper':<18} {'Ours':<18} {'Diff':<10} {'Within ±3%?':<12}"
    print(header)
    print("-" * 90)

    for task in TASK_ORDER:
        paper = PAPER_RESULTS.get(task, {})
        ours = stats.get(task, {})

        if paper and ours:
            paper_str = f"{paper['mean']:.2f} ± {paper['std']:.2f}"
            ours_str = f"{ours['mean']:.2f} ± {ours['std']:.2f}"
            diff = ours['mean'] - paper['mean']
            diff_str = f"{diff:+.2f}"
            within = abs(diff) <= 3.0
            status = "YES" if within else "NO"
            print(f"{task:<12} {paper_str:<18} {ours_str:<18} {diff_str:<10} {status:<12}")
        elif paper:
            paper_str = f"{paper['mean']:.2f} ± {paper['std']:.2f}"
            print(f"{task:<12} {paper_str:<18} {'N/A':<18} {'N/A':<10} {'N/A':<12}")

    print("-" * 90)

    if "average" in stats:
        avg_diff = abs(stats["average"]["mean"] - PAPER_RESULTS["average"]["mean"])
        print(f"\nOverall average difference: {avg_diff:.2f}%")
        print(f"Target: ±3.0% → {'PASS' if avg_diff <= 3.0 else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FedLEASE experiment results")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Root output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                        help="Seeds to analyze")
    parser.add_argument("--model_name", type=str, default="roberta-large")
    parser.add_argument("--num_clients", type=int, default=16)
    args = parser.parse_args()

    print(f"Analyzing results from: {args.output_dir}")
    print(f"Seeds: {args.seeds}")
    print()

    results = load_results(args.output_dir, args.seeds, args.model_name, args.num_clients)

    if not results:
        print("\nNo results found yet. Experiments may still be running.")
        return

    print(f"\nLoaded results for {len(results)} seeds: {sorted(results.keys())}")

    stats = compute_statistics(results)
    print_comparison_table(stats)

    # Save summary
    summary = {
        "seeds": sorted(results.keys()),
        "per_seed_results": {str(k): v for k, v in results.items()},
        "statistics": {k: {"mean": v["mean"], "std": v["std"], "n": v["n"]}
                      for k, v in stats.items()},
        "paper_results": PAPER_RESULTS,
    }
    summary_path = os.path.join(args.output_dir, "reproduction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
