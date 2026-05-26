#!/usr/bin/env python
"""Head-portability matrix.

Evaluates every off-diagonal cell (source expert e_c on target task t) under
*multiple* classification heads, to test whether WOS-trained experts produce
representations readable by task heads they never co-trained with.

For each target cluster we evaluate, per cell:
  - home            : the source client's own head (the ~48 floor)
  - client_<cid>    : each individual client head of the target cluster, one at
                      a time (no averaging)
  - cluster_average : the average of *all* target-cluster client heads
                      (= the existing head_swap_eval cluster_average metric)
  - loco_<cid>      : leave-one-client-out averages (average of the target
                      cluster minus client cid)

If WOS lifts off-diagonal accuracy under *every* individual / LOCO head while
the FedAvg baseline stays low, the lift is a property of the expert
representation (head portability), not an artifact of one lucky head average.

This script does NOT modify the locked scripts/eval/head_swap_eval.py; it imports
its helpers. Run with the same protocol as the headline metric:
  --force-eval-mode home_cluster_only
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

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
    _average_classifier_states,
    _build_client,
    _build_client_cluster_map,
    _build_cluster_metadata,
    _build_task_cluster_map,
    _client_params,
    _extract_classifier_state_from_model,
    _extract_classifier_state_from_params,
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
    _routing_override_context,
    _task_order,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Head-portability matrix: evaluate each off-diagonal cell under "
            "every individual / leave-one-out / averaged target-cluster head."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Run directory containing final_params.pt and proposed_m2/training_history.json",
    )
    parser.add_argument("--output", default=None, help="Path for results JSON")
    parser.add_argument(
        "--force-eval-mode",
        choices=("auto", "natural", "home_cluster_only", "cluster_only"),
        default="home_cluster_only",
        help="Eval forward mode override; headline protocol uses home_cluster_only.",
    )
    parser.add_argument(
        "--force-routing",
        choices=("auto", "target_cluster"),
        default="auto",
        help="Inference-time routing override (headline protocol leaves it auto).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional override when model_name cannot be recovered from the checkpoint",
    )
    parser.add_argument(
        "--limit-clients",
        type=int,
        default=None,
        help="Debug only: evaluate the first N source clients",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Debug only: cap each validation task to the first N samples",
    )
    return parser.parse_args()


def _build_portability_heads(final_params, lora_client_map):
    """Per target cluster, build every head variant we evaluate.

    Returns {cluster_id: {variant_name: head_state}} with variants:
      client_<cid>     individual head of each cluster member
      cluster_average  average of all members
      loco_<cid>       average of all members except cid (only when >=2 members)
    """
    heads = {}
    for cluster_id, client_ids in sorted(lora_client_map.items()):
        if not client_ids:
            raise ValueError(f"Cluster {cluster_id} has no member clients")
        client_ids = [int(c) for c in client_ids]
        per_client = {
            cid: _extract_classifier_state_from_params(
                _client_params(final_params, cid), cluster_id
            )
            for cid in client_ids
        }
        variants = {f"client_{cid}": state for cid, state in per_client.items()}
        variants["cluster_average"] = _average_classifier_states(
            list(per_client.values()), cluster_id
        )
        if len(client_ids) >= 2:
            for cid in client_ids:
                others = [per_client[o] for o in client_ids if o != cid]
                variants[f"loco_{cid}"] = _average_classifier_states(others, cluster_id)
        heads[int(cluster_id)] = variants
    return heads


def _evaluate_portability(
    final_params,
    task_info,
    lora_client_map,
    client_to_cluster,
    cluster_metadata,
    portability_heads,
    task_to_cluster,
    task_names,
    task_eval_datasets,
    task_metadata,
    tokenizer,
    config,
    selected_clients,
    force_eval_mode,
    force_routing,
):
    # per_cell[str(source_client)][target_task] = {head_variant: accuracy}
    per_cell = {}

    for client_id in tqdm(selected_clients, desc="portability source clients"):
        source_num_labels = int(task_info[client_id]["num_labels"])
        client = _build_client(
            client_id=client_id,
            task_info=task_info,
            tokenizer=tokenizer,
            config=config,
            num_clients=len(task_info),
            client_to_cluster=client_to_cluster,
        )
        client.load_model()
        _apply_eval_mode_override(client, force_eval_mode)
        client.load_params(_client_params(final_params, client_id))
        client.cache_path = str(
            Path(config["cache_path"]) / f"head_portability__{force_eval_mode}_tmp"
        )

        per_cell[str(client_id)] = {}
        try:
            for target_task in task_names:
                target_num_labels = int(task_metadata[target_task]["num_labels"])
                if source_num_labels != target_num_labels:
                    per_cell[str(client_id)][target_task] = None
                    continue

                target_cluster = task_to_cluster[target_task]
                cell = {}

                # home head: the source client's own classifier
                with _routing_override_context(
                    client.local_model, force_routing, target_cluster
                ):
                    metrics = client.evaluate_on_dataset(
                        task_eval_datasets[target_task],
                        num_labels=target_num_labels,
                        dataset_name=f"{target_task}_home",
                    )
                cell["home"] = round(_accuracy_from_metrics(metrics), 4)

                home_head = _extract_classifier_state_from_model(client.local_model)
                try:
                    for variant, head_state in portability_heads[target_cluster].items():
                        _load_classifier_state(
                            client.local_model,
                            head_state,
                            context=(
                                f"client {client_id}, target {target_task}, "
                                f"head {variant}"
                            ),
                        )
                        with _routing_override_context(
                            client.local_model, force_routing, target_cluster
                        ):
                            metrics = client.evaluate_on_dataset(
                                task_eval_datasets[target_task],
                                num_labels=target_num_labels,
                                dataset_name=f"{target_task}_{variant}",
                            )
                        cell[variant] = round(_accuracy_from_metrics(metrics), 4)
                finally:
                    _load_classifier_state(
                        client.local_model,
                        home_head,
                        context=f"restore client {client_id} home head",
                    )

                per_cell[str(client_id)][target_task] = cell
        finally:
            _free_client_model(client)

    return per_cell


def _aggregate(per_cell, task_info, client_to_cluster, cluster_metadata):
    """Aggregate per-cell scores into in-dist / off-diag means per head variant."""
    label_to_tasks = {
        meta["label"]: set(meta["tasks"]) for meta in cluster_metadata.values()
    }
    # head variant -> {"in": [...], "off": [...]}
    buckets = defaultdict(lambda: {"in": [], "off": []})

    for client_id_raw, task_scores in per_cell.items():
        client_id = int(client_id_raw)
        cluster_label = cluster_metadata[client_to_cluster[client_id]]["label"]
        home_tasks = label_to_tasks.get(cluster_label, set())
        for target_task, cell in task_scores.items():
            if cell is None:
                continue
            axis = "in" if target_task in home_tasks else "off"
            for variant, score in cell.items():
                buckets[variant][axis].append(float(score))

    summary = {}
    for variant, axes in sorted(buckets.items()):
        entry = {}
        for axis in ("in", "off"):
            values = axes[axis]
            entry[f"{axis}_dist" if axis == "in" else "offdiag"] = (
                round(sum(values) / len(values), 4) if values else math.nan
            )
            entry[f"n_{axis}_cells"] = len(values)
        summary[variant] = entry
    return summary


def _portability_offdiag_stats(summary, cluster_metadata):
    """Collapse individual client_* and loco_* variants into mean / min off-diag,
    so the paper can report a single 'portability is robust across heads' number.
    """
    client_offdiags = [
        v["offdiag"]
        for name, v in summary.items()
        if name.startswith("client_") and not math.isnan(v["offdiag"])
    ]
    loco_offdiags = [
        v["offdiag"]
        for name, v in summary.items()
        if name.startswith("loco_") and not math.isnan(v["offdiag"])
    ]
    stats = {}
    if client_offdiags:
        stats["individual_head_offdiag_mean"] = round(
            sum(client_offdiags) / len(client_offdiags), 4
        )
        stats["individual_head_offdiag_min"] = round(min(client_offdiags), 4)
        stats["individual_head_offdiag_max"] = round(max(client_offdiags), 4)
    if loco_offdiags:
        stats["loco_head_offdiag_mean"] = round(
            sum(loco_offdiags) / len(loco_offdiags), 4
        )
        stats["loco_head_offdiag_min"] = round(min(loco_offdiags), 4)
    if "home" in summary:
        stats["home_head_offdiag"] = summary["home"]["offdiag"]
    if "cluster_average" in summary:
        stats["cluster_average_offdiag"] = summary["cluster_average"]["offdiag"]
    return stats


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
        history.get("lora_client_map"), len(task_info)
    )
    client_to_cluster = _build_client_cluster_map(lora_client_map, len(task_info))
    cluster_metadata = _build_cluster_metadata(lora_client_map, task_info)
    config = _load_run_config(run_dir, history, latest_checkpoint, args)

    if config["fedrod_dual_head"]:
        raise NotImplementedError(
            "head_portability_matrix.py supports non-FedRoD checkpoints only."
        )

    selected_clients = sorted(task_info)
    if args.limit_clients is not None:
        selected_clients = selected_clients[: args.limit_clients]

    print(f"Run directory: {run_dir}")
    print(f"Params source: {params_source}")
    print(f"Model: {config['model_name']}")
    print(f"Force eval mode: {args.force_eval_mode}")
    print(f"Force routing: {args.force_routing}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], cache_dir=config["cache_path"]
    )
    task_names = _task_order(task_info)
    task_names, task_eval_datasets, task_metadata = _prepare_task_datasets(
        task_names, tokenizer, limit_tasks=None, limit_samples=args.limit_samples
    )

    portability_heads = _build_portability_heads(final_params, lora_client_map)
    task_to_cluster = _build_task_cluster_map(lora_client_map, task_info)

    per_cell = _evaluate_portability(
        final_params=final_params,
        task_info=task_info,
        lora_client_map=lora_client_map,
        client_to_cluster=client_to_cluster,
        cluster_metadata=cluster_metadata,
        portability_heads=portability_heads,
        task_to_cluster=task_to_cluster,
        task_names=task_names,
        task_eval_datasets=task_eval_datasets,
        task_metadata=task_metadata,
        tokenizer=tokenizer,
        config=config,
        selected_clients=selected_clients,
        force_eval_mode=args.force_eval_mode,
        force_routing=args.force_routing,
    )

    summary = _aggregate(per_cell, task_info, client_to_cluster, cluster_metadata)
    portability_stats = _portability_offdiag_stats(summary, cluster_metadata)

    print("\nHead-portability summary (off-diagonal accuracy by head variant):")
    for variant, entry in summary.items():
        print(
            f"  {variant:20s} in={entry['in_dist']:.4f}  off={entry['offdiag']:.4f}"
            f"  (n_off={entry['n_off_cells']})"
        )
    print("\nPortability stats:")
    for key, value in portability_stats.items():
        print(f"  {key}: {value}")

    results = {
        "checkpoint_dir": str(run_dir),
        "params_source": params_source,
        "history_path": str(history_path),
        "force_eval_mode": args.force_eval_mode,
        "forced_routing": args.force_routing,
        "task_order": task_names,
        "cluster_metadata": {
            str(cid): meta for cid, meta in cluster_metadata.items()
        },
        "task_to_cluster": {
            t: int(c) for t, c in sorted(task_to_cluster.items())
        },
        "head_variants_per_cluster": {
            str(cid): sorted(variants)
            for cid, variants in portability_heads.items()
        },
        "summary_by_head": summary,
        "portability_stats": portability_stats,
        "per_cell": per_cell,
    }

    output_path = (
        Path(args.output) if args.output else run_dir / "head_portability_matrix.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
