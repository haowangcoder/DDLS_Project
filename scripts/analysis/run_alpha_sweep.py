#!/usr/bin/env python

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]  # repo root (scripts/analysis/<file>)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import Client
from utils import load_full_task_validation_datasets


DEFAULT_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc alpha sweep for clustered LoRA experts")
    parser.add_argument("--params_dir", type=str, required=True, help="Run directory containing final_params.pt")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS, help="Alpha values to evaluate")
    return parser.parse_args()


def _resolve_training_history_path(params_dir: Path) -> Path:
    candidates = [
        params_dir / "proposed_m2" / "training_history.json",
        params_dir / "training_history.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find training_history.json under {params_dir}")


def _load_final_params(params_dir: Path):
    params_path = params_dir / "final_params.pt"
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find final_params.pt under {params_dir}")

    payload = torch.load(params_path, map_location="cpu")
    if isinstance(payload, dict) and "aggregated_params" in payload:
        return payload["aggregated_params"]
    return payload


def _normalize_task_info(raw_task_info):
    return {int(client_id): info for client_id, info in raw_task_info.items()}


def _normalize_lora_client_map(raw_lora_client_map):
    if raw_lora_client_map is None:
        return None
    return {
        int(cluster_id): [int(client_id) for client_id in client_ids]
        for cluster_id, client_ids in raw_lora_client_map.items()
    }


def _build_client_cluster_map(lora_client_map, num_clients):
    if not lora_client_map:
        return {client_id: 0 for client_id in range(num_clients)}

    client_to_cluster = {}
    for cluster_id, cluster_clients in lora_client_map.items():
        for client_id in cluster_clients:
            client_to_cluster[int(client_id)] = int(cluster_id)

    for client_id in range(num_clients):
        client_to_cluster.setdefault(client_id, 0)
    return client_to_cluster


def _b_template(param_name: str) -> str:
    return re.sub(r"lora_B\d+", "lora_B{}", param_name)


def _b_name_for_cluster(template: str, cluster_idx: int) -> str:
    return template.replace("{}", str(cluster_idx))


def _is_assigned_b_name(param_name: str, cluster_idx: int) -> bool:
    return re.search(rf"lora_B{cluster_idx}(?!\d)", param_name) is not None


def _compute_universal_b(final_params, lora_client_map):
    if not lora_client_map:
        raise ValueError("Alpha sweep requires lora_client_map from clustered training")

    cluster_states = {
        cluster_id: final_params[cluster_clients[0]]
        for cluster_id, cluster_clients in sorted(lora_client_map.items())
        if cluster_clients
    }

    universal_b = {}
    reference_state = next(iter(cluster_states.values()))
    for param_name in reference_state:
        if "lora_B" not in param_name:
            continue

        template = _b_template(param_name)
        tensors = []
        for cluster_id in cluster_states:
            candidate_name = _b_name_for_cluster(template, cluster_id)
            candidate_value = cluster_states[cluster_id].get(candidate_name)
            if candidate_value is not None:
                tensors.append(candidate_value.float())

        if tensors:
            universal_b[template] = torch.stack(tensors).mean(dim=0)

    return universal_b


def _build_mixed_b_updates(base_params, cluster_idx, universal_b, alpha):
    updates = {}
    for param_name, param_value in base_params.items():
        if not _is_assigned_b_name(param_name, cluster_idx):
            continue

        template = _b_template(param_name)
        if template not in universal_b:
            continue

        mixed_value = alpha * param_value.float() + (1.0 - alpha) * universal_b[template]
        updates[param_name] = mixed_value.to(dtype=param_value.dtype)

    return updates


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    params_dir = Path(args.params_dir).resolve()
    history_path = _resolve_training_history_path(params_dir)
    final_params = _load_final_params(params_dir)

    with open(history_path, "r") as f:
        history = json.load(f)

    task_info = _normalize_task_info(history["task_info"])
    lora_client_map = _normalize_lora_client_map(history["lora_client_map"])
    if history.get("universal_expert"):
        print("Warning: run already includes a trained universal expert; alpha sweep will still use post-hoc cluster-B mixing.")

    final_lora_n = history.get("final_lora_n") or history.get("optimal_n_clusters")
    final_adaptive = history.get("final_adaptive", True)

    task_names = list(dict.fromkeys(task_info[client_id]["task_name"] for client_id in sorted(task_info)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(params_dir))
    task_eval_datasets, task_metadata = load_full_task_validation_datasets(task_names, tokenizer)

    client_to_cluster = _build_client_cluster_map(lora_client_map, len(final_params))
    universal_b = _compute_universal_b(final_params, lora_client_map)

    results = {
        "alphas": [round(alpha, 4) for alpha in args.alphas],
        "task_order": task_names,
        "per_client": {},
    }
    average_by_source_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for client_id in sorted(task_info):
        client_cluster = client_to_cluster[client_id]
        source_task = task_info[client_id]["task_name"]
        num_labels = task_info[client_id]["num_labels"]
        base_params = final_params[client_id]

        client = Client(
            client_id=client_id,
            task_name=source_task,
            tokenizer=tokenizer,
            model_name=args.model_name,
            num_clients=len(final_params),
            rank=history.get("rank", 4),
            lora_n=final_lora_n,
            adaptive=final_adaptive,
            cache_path=str(params_dir),
            idx=client_cluster,
            universal_idx=history.get("universal_idx"),
        )
        client.set_dataset({"validation": task_eval_datasets[source_task]}, num_labels)
        client.load_model()
        client.load_params(base_params)

        target_results = {task_name: {} for task_name in task_names}
        for alpha in args.alphas:
            alpha_key = f"{alpha:.1f}"
            client.load_params(_build_mixed_b_updates(base_params, client_cluster, universal_b, alpha))

            for target_task in task_names:
                metrics = client.evaluate_on_dataset(
                    task_eval_datasets[target_task],
                    num_labels=task_metadata[target_task]["num_labels"],
                    dataset_name=f"{target_task}_validation",
                )
                accuracy = round(metrics.get("eval_accuracy", metrics.get("accuracy", 0.0)) * 100, 4)
                target_results[target_task][alpha_key] = accuracy
                average_by_source_task[source_task][target_task][alpha_key].append(accuracy)

        client.unload_model()

        results["per_client"][str(client_id)] = {
            "source_task": source_task,
            "source_cluster": client_cluster,
            "target_tasks": target_results,
        }

    results["average_by_source_task"] = {
        source_task: {
            target_task: {
                alpha_key: round(sum(values) / len(values), 4)
                for alpha_key, values in sorted(alpha_scores.items())
            }
            for target_task, alpha_scores in sorted(target_scores.items())
        }
        for source_task, target_scores in sorted(average_by_source_task.items())
    }

    output_path = params_dir / "alpha_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved alpha sweep results to {output_path}")


if __name__ == "__main__":
    main()
