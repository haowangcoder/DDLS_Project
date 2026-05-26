#!/usr/bin/env python
"""Evaluate offline uniform delta-merge of FedLEASE cluster experts.

This script measures three in-distribution modes for one checkpoint:
  - routed: oracle home-cluster expert, weight 1.0
  - home_scaled: home expert only, weight 1/k
  - merged: all k cluster experts, uniform weight 1/k

It intentionally imports helpers from the locked head_swap_eval.py rather than
duplicating checkpoint, dataset, client, and classifier-head logic.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]  # repo root (scripts/eval/<file>)
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from head_swap_eval import (  # noqa: E402  (path bootstrap must run first)
    _accuracy_from_metrics,
    _apply_eval_mode_override,
    _build_client,
    _build_client_cluster_map,
    _build_cluster_heads,
    _build_cluster_metadata,
    _build_task_cluster_map,
    _client_params,
    _extract_classifier_state_from_model,
    _free_client_model,
    _load_classifier_state,
    _load_final_params,
    _load_latest_checkpoint,
    _load_run_config,
    _normalize_lora_client_map,
    _normalize_task_info,
    _prepare_task_datasets,
    _resolve_training_history_path,
    _round_or_nan,
    _task_order,
)


_EXPERT_KEY_RE = re.compile(r"(?:^|\.)lora_[AB]\d+(?:\.|$)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate routed, home-scaled, and uniformly merged FedLEASE cluster "
            "experts on one checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Run directory containing final_params.pt and proposed_m2/training_history.json",
    )
    parser.add_argument("--output", default=None, help="Path for results JSON")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional override when model_name cannot be recovered from the checkpoint",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Experiment arm label to echo verbatim into the JSON",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Experiment seed label to echo verbatim into the JSON",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit task-name list to evaluate; default is all trained tasks",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Smoke only: cap each validation task to the first N samples",
    )
    return parser.parse_args()


def _expert_tensors(params):
    return {
        name: value.detach().cpu()
        for name, value in params.items()
        if isinstance(value, torch.Tensor) and _EXPERT_KEY_RE.search(name)
    }


def _assert_shared_experts(final_params, n_clients):
    """Fail unless all client payloads hold identical LoRA expert tensors."""
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive, got {n_clients}")

    reference_client = 0
    reference = _expert_tensors(_client_params(final_params, reference_client))
    if not reference:
        raise ValueError(
            f"Client {reference_client} has no lora_A*/lora_B* tensors in final params"
        )

    reference_keys = set(reference)
    for client_id in range(1, int(n_clients)):
        current = _expert_tensors(_client_params(final_params, client_id))
        current_keys = set(current)
        if current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys)
            extra = sorted(current_keys - reference_keys)
            raise ValueError(
                f"Client {client_id} expert key mismatch vs client {reference_client}: "
                f"missing={missing}, extra={extra}"
            )

        for name in sorted(reference_keys):
            expected = reference[name]
            actual = current[name]
            if expected.dtype != actual.dtype:
                raise ValueError(
                    f"Expert dtype mismatch for {name}: client {reference_client} "
                    f"{expected.dtype} != client {client_id} {actual.dtype}"
                )
            if tuple(expected.shape) != tuple(actual.shape):
                raise ValueError(
                    f"Expert shape mismatch for {name}: client {reference_client} "
                    f"{tuple(expected.shape)} != client {client_id} {tuple(actual.shape)}"
                )
            if not torch.equal(expected, actual):
                raise ValueError(
                    f"Expert tensor mismatch for {name}: client {client_id} differs "
                    f"from client {reference_client}"
                )


def _iter_visa_modules(model):
    for module in model.modules():
        if getattr(module, "universal_idx", None) is not None and hasattr(
            module, "lora_route"
        ):
            yield module


def _set_visa_weights(model, weight_vec):
    if not isinstance(weight_vec, torch.Tensor):
        raise TypeError(f"weight_vec must be a torch.Tensor, got {type(weight_vec)}")
    if weight_vec.dim() != 1:
        raise ValueError(f"weight_vec must be 1-D, got shape {tuple(weight_vec.shape)}")

    module_count = 0
    for module in _iter_visa_modules(model):
        expected = int(module.lora_num) - 1
        if weight_vec.numel() != expected:
            raise ValueError(
                f"visa weight length {weight_vec.numel()} does not match module "
                f"cluster count {expected}"
            )
        route_weight = module.lora_route.weight
        module._visa_cluster_weights = (
            weight_vec.detach()
            .to(device=route_weight.device, dtype=route_weight.dtype)
            .clone()
        )
        module_count += 1

    if module_count == 0:
        raise ValueError("No modules with universal_idx and lora_route found")


def _clear_visa_weights(model):
    for module in _iter_visa_modules(model):
        if hasattr(module, "_visa_cluster_weights"):
            delattr(module, "_visa_cluster_weights")


def _mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else math.nan


def _evaluate_task_modes(
    *,
    final_params,
    canonical_client,
    task_name,
    cluster_id,
    cluster_heads,
    task_dataset,
    num_labels,
    task_info,
    tokenizer,
    config,
    num_clients,
    client_to_cluster,
    lora_client_map,
    num_cluster_experts,
    run_dir,
):
    representatives = lora_client_map.get(cluster_id)
    if not representatives:
        raise ValueError(f"Cluster {cluster_id} has no representative clients")

    representative = int(representatives[0])
    client = _build_client(
        client_id=representative,
        task_info=task_info,
        tokenizer=tokenizer,
        config=config,
        num_clients=num_clients,
        client_to_cluster=client_to_cluster,
    )
    client.load_model()
    client.cache_path = str(run_dir / "expert_merge_eval_tmp")

    try:
        client.load_params(_client_params(final_params, canonical_client))
        _load_classifier_state(
            client.local_model,
            cluster_heads[cluster_id],
            context=f"task {task_name}, cluster {cluster_id}, cluster_average head",
        )

        # Imported per spec; the returned state also checks that a classifier is present.
        _extract_classifier_state_from_model(client.local_model)

        _apply_eval_mode_override(client, "home_cluster_only")
        metrics = client.evaluate_on_dataset(
            task_dataset,
            num_labels=num_labels,
            dataset_name=f"{task_name}_routed",
        )
        routed_acc = _accuracy_from_metrics(metrics)

        home_vec = torch.zeros(num_cluster_experts, dtype=torch.float32)
        if cluster_id < 0 or cluster_id >= num_cluster_experts:
            raise ValueError(
                f"Task {task_name} cluster {cluster_id} is outside "
                f"non-universal expert range [0, {num_cluster_experts})"
            )
        home_vec[cluster_id] = 1.0 / float(num_cluster_experts)
        _set_visa_weights(client.local_model, home_vec)
        _apply_eval_mode_override(client, "non_home_visa")
        metrics = client.evaluate_on_dataset(
            task_dataset,
            num_labels=num_labels,
            dataset_name=f"{task_name}_home_scaled",
        )
        home_scaled_acc = _accuracy_from_metrics(metrics)

        merged_vec = torch.full(
            (num_cluster_experts,),
            1.0 / float(num_cluster_experts),
            dtype=torch.float32,
        )
        _set_visa_weights(client.local_model, merged_vec)
        metrics = client.evaluate_on_dataset(
            task_dataset,
            num_labels=num_labels,
            dataset_name=f"{task_name}_merged",
        )
        merged_acc = _accuracy_from_metrics(metrics)
    finally:
        if client.local_model is not None:
            _clear_visa_weights(client.local_model)
        _free_client_model(client)

    return {
        "cluster": int(cluster_id),
        "routed_acc": _round_or_nan(routed_acc),
        "home_scaled_acc": _round_or_nan(home_scaled_acc),
        "merged_acc": _round_or_nan(merged_acc),
        "retention": _round_or_nan(merged_acc - routed_acc),
    }


def main():
    args = parse_args()
    run_dir = Path(args.checkpoint_dir).resolve()

    history_path = _resolve_training_history_path(run_dir)
    with history_path.open() as f:
        history = json.load(f)

    latest_checkpoint = None
    try:
        final_params, params_source = _load_final_params(run_dir, latest_checkpoint)
    except FileNotFoundError:
        latest_checkpoint = _load_latest_checkpoint(run_dir)
        final_params, params_source = _load_final_params(run_dir, latest_checkpoint)

    task_info = _normalize_task_info(history["task_info"])
    lora_client_map = _normalize_lora_client_map(
        history.get("lora_client_map"),
        len(task_info),
    )
    client_to_cluster = _build_client_cluster_map(lora_client_map, len(task_info))
    cluster_metadata = _build_cluster_metadata(lora_client_map, task_info)
    config = _load_run_config(run_dir, history, latest_checkpoint, args)

    if config["fedrod_dual_head"]:
        raise NotImplementedError(
            "expert_merge_eval.py supports non-FedRoD checkpoints only."
        )

    _assert_shared_experts(final_params, len(task_info))

    if config["universal_idx"] is None:
        raise ValueError("Expert merge requires a universal_idx in the checkpoint config")
    num_cluster_experts = int(config["lora_n"]) - 1
    if num_cluster_experts <= 0:
        raise ValueError(
            f"Expected at least one cluster expert, got lora_n={config['lora_n']}"
        )
    if int(config["universal_idx"]) != num_cluster_experts:
        raise ValueError(
            "Expert merge assumes non-universal experts occupy indices 0..k-1; "
            f"got lora_n={config['lora_n']} and universal_idx={config['universal_idx']}"
        )

    canonical_client = sorted(task_info)[0]
    all_tasks = _task_order(task_info)
    task_names = list(args.tasks) if args.tasks is not None else list(all_tasks)
    unknown_tasks = [task_name for task_name in task_names if task_name not in all_tasks]
    if unknown_tasks:
        raise ValueError(
            f"Unknown task(s) requested: {unknown_tasks}. Available tasks: {all_tasks}"
        )

    print(f"Run directory: {run_dir}")
    print(f"Params source: {params_source}")
    print(f"History: {history_path}")
    print(f"Model: {config['model_name']}")
    print(
        f"LoRA experts: {config['lora_n']} "
        f"(cluster experts={num_cluster_experts}, universal_idx={config['universal_idx']})"
    )
    print(f"Canonical expert client: {canonical_client}")
    print(f"Tasks: {', '.join(task_names)}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        cache_dir=config["cache_path"],
    )
    task_names, task_eval_datasets, task_metadata = _prepare_task_datasets(
        task_names,
        tokenizer,
        limit_tasks=None,
        limit_samples=args.limit_samples,
    )

    cluster_heads = _build_cluster_heads(
        final_params,
        lora_client_map,
        head_source="cluster_average",
    )
    task_to_cluster = _build_task_cluster_map(lora_client_map, task_info)

    per_task = {}
    for task_name in tqdm(task_names, desc="expert-merge tasks"):
        if task_name not in task_to_cluster:
            raise ValueError(f"Task {task_name} has no cluster assignment")
        if task_name not in task_eval_datasets:
            raise ValueError(f"Task {task_name} validation dataset was not loaded")
        if task_name not in task_metadata:
            raise ValueError(f"Task {task_name} metadata was not loaded")

        cluster_id = int(task_to_cluster[task_name])
        if cluster_id not in cluster_heads:
            raise ValueError(f"Cluster {cluster_id} has no cluster-average head")

        per_task[task_name] = _evaluate_task_modes(
            final_params=final_params,
            canonical_client=canonical_client,
            task_name=task_name,
            cluster_id=cluster_id,
            cluster_heads=cluster_heads,
            task_dataset=task_eval_datasets[task_name],
            num_labels=int(task_metadata[task_name]["num_labels"]),
            task_info=task_info,
            tokenizer=tokenizer,
            config=config,
            num_clients=len(task_info),
            client_to_cluster=client_to_cluster,
            lora_client_map=lora_client_map,
            num_cluster_experts=num_cluster_experts,
            run_dir=run_dir,
        )

    routed_values = [entry["routed_acc"] for entry in per_task.values()]
    home_scaled_values = [entry["home_scaled_acc"] for entry in per_task.values()]
    merged_values = [entry["merged_acc"] for entry in per_task.values()]
    retention_values = [entry["retention"] for entry in per_task.values()]
    summary = {
        "routed_in_dist": _round_or_nan(_mean(routed_values)),
        "home_scaled_in_dist": _round_or_nan(_mean(home_scaled_values)),
        "merged_in_dist": _round_or_nan(_mean(merged_values)),
        "retention": _round_or_nan(_mean(retention_values)),
        "n_tasks": len(per_task),
    }

    print("\nExpert-merge summary:")
    print(f"  routed:      {summary['routed_in_dist']:.4f}")
    print(f"  home_scaled: {summary['home_scaled_in_dist']:.4f}")
    print(f"  merged:      {summary['merged_in_dist']:.4f}")
    print(f"  retention:   {summary['retention']:+.4f}")

    results = {
        "checkpoint_dir": str(run_dir),
        "params_source": params_source,
        "history_path": str(history_path),
        "arm": args.arm,
        "seed": args.seed,
        "num_cluster_experts": num_cluster_experts,
        "task_order": task_names,
        "cluster_metadata": {
            str(cluster_id): meta for cluster_id, meta in cluster_metadata.items()
        },
        "task_to_cluster": {
            task_name: int(cluster_id)
            for task_name, cluster_id in sorted(task_to_cluster.items())
        },
        "config": {
            "model_name": config["model_name"],
            "rank": config["rank"],
            "lora_n": config["lora_n"],
            "universal_idx": config["universal_idx"],
            "additive_residual": config["additive_residual"],
            "visa_coeff": config["visa_coeff"],
        },
        "per_task": per_task,
        "summary": summary,
    }

    output_path = Path(args.output) if args.output else run_dir / "expert_merge_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
