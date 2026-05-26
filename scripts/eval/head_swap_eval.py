#!/usr/bin/env python

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding


ROOT = Path(__file__).resolve().parents[2]  # repo root (scripts/eval/<file>)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import Client
from peft.tuners.lora import uemd_forward_mode
from utils import load_full_task_validation_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FedLEASE cross-task accuracy with home vs target cluster classifier heads."
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
        default="auto",
        help=(
            "Override Client._eval_forward_context during eval. auto keeps Client behavior; "
            "natural uses nullcontext; home_cluster_only/cluster_only force uemd_forward_mode."
        ),
    )
    parser.add_argument(
        "--head-source",
        choices=(
            "first_client",
            "cluster_average",
            "shuffled_other_cluster",
            "knn_predicted",
        ),
        default="first_client",
        help=(
            "Classifier head source for the target-head protocol. first_client uses the "
            "cluster representative, cluster_average averages all cluster client heads, "
            "shuffled_other_cluster uses a deterministic same-num-label other cluster, "
            "and knn_predicted adds a task-blind KNN-head protocol using frozen-base CLS prototypes."
        ),
    )
    parser.add_argument(
        "--force-routing",
        choices=("auto", "target_cluster"),
        default="auto",
        help=(
            "Inference-time routing override. auto leaves routing unchanged; "
            "target_cluster forces one-hot routing to cluster_of(target_task)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional override when model_name cannot be recovered from the run checkpoint",
    )
    parser.add_argument(
        "--limit-clients",
        type=int,
        default=None,
        help="Debug only: evaluate the first N source clients",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Debug only: evaluate the first N target tasks",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Debug only: cap each validation task to the first N samples",
    )
    parser.add_argument(
        "--prototype-samples",
        type=int,
        default=200,
        help="Number of validation examples per cluster used to build frozen-base KNN prototypes",
    )
    parser.add_argument(
        "--knn-batch-size",
        type=int,
        default=256,
        help="Batch size for frozen-base prototype extraction and KNN prediction",
    )
    return parser.parse_args()


def _resolve_training_history_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "proposed_m2" / "training_history.json",
        run_dir / "training_history.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find training_history.json under {run_dir}")


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_latest_checkpoint(run_dir: Path):
    pointer_path = run_dir / "checkpoints" / "latest.json"
    if not pointer_path.exists():
        return None

    with pointer_path.open() as f:
        pointer = json.load(f)
    checkpoint_path = pointer.get("checkpoint_path") or pointer.get("path")
    if not checkpoint_path:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    return _torch_load_cpu(checkpoint_path)


def _load_final_params(run_dir: Path, latest_checkpoint=None):
    params_path = run_dir / "final_params.pt"
    if params_path.exists():
        payload = _torch_load_cpu(params_path)
        if isinstance(payload, dict) and "aggregated_params" in payload:
            return payload["aggregated_params"], str(params_path)
        return payload, str(params_path)

    if latest_checkpoint is not None:
        fed_state = latest_checkpoint.get("fed_state") or {}
        aggregated_params = fed_state.get("aggregated_params")
        if aggregated_params is not None:
            return aggregated_params, str(run_dir / "checkpoints" / "latest.json")

    raise FileNotFoundError(
        f"Could not find final_params.pt under {run_dir}, and latest checkpoint has no aggregated_params"
    )


def _normalize_task_info(raw_task_info):
    return {
        int(client_id): {
            "task_name": info["task_name"],
            "num_labels": int(info["num_labels"]),
        }
        for client_id, info in raw_task_info.items()
    }


def _normalize_lora_client_map(raw_lora_client_map, num_clients):
    if raw_lora_client_map is None:
        return {0: list(range(num_clients))}
    return {
        int(cluster_id): [int(client_id) for client_id in client_ids]
        for cluster_id, client_ids in raw_lora_client_map.items()
    }


def _build_client_cluster_map(lora_client_map, num_clients):
    client_to_cluster = {}
    for cluster_id, client_ids in lora_client_map.items():
        for client_id in client_ids:
            client_to_cluster[int(client_id)] = int(cluster_id)
    for client_id in range(num_clients):
        client_to_cluster.setdefault(client_id, 0)
    return client_to_cluster


def _build_cluster_metadata(lora_client_map, task_info):
    metadata = {}
    for cluster_id, client_ids in sorted(lora_client_map.items()):
        tasks = sorted({task_info[client_id]["task_name"] for client_id in client_ids})
        num_labels = sorted({task_info[client_id]["num_labels"] for client_id in client_ids})
        label = tasks[0] if len(tasks) == 1 else f"cluster_{cluster_id}"
        metadata[int(cluster_id)] = {
            "label": label,
            "clients": [int(client_id) for client_id in client_ids],
            "tasks": tasks,
            "num_labels": num_labels[0] if len(num_labels) == 1 else None,
        }
    return metadata


def _task_order(task_info):
    return list(
        dict.fromkeys(
            task_info[client_id]["task_name"] for client_id in sorted(task_info)
        )
    )


def _coalesce_float(*values, default=0.0):
    for value in values:
        if value is not None:
            return float(value)
    return float(default)


def _coalesce_bool(*values, default=False):
    for value in values:
        if value is not None:
            return bool(value)
    return bool(default)


def _load_run_config(run_dir: Path, history, latest_checkpoint, args):
    checkpoint_args = (latest_checkpoint or {}).get("args") or {}
    model_name = args.model_name or checkpoint_args.get("model_name") or "roberta-large"
    optimal_n_clusters = history.get("optimal_n_clusters") or checkpoint_args.get("max_clusters")
    universal_expert = _coalesce_bool(
        history.get("universal_expert"),
        checkpoint_args.get("universal_expert"),
        default=False,
    )

    if history.get("final_lora_n") is not None:
        lora_n = int(history["final_lora_n"])
    elif optimal_n_clusters is not None:
        lora_n = int(optimal_n_clusters) + (1 if universal_expert else 0)
    else:
        lora_n = int(checkpoint_args.get("lora_n") or 1)

    return {
        "model_name": model_name,
        "rank": int(history.get("rank") or checkpoint_args.get("rank") or 4),
        "lora_n": lora_n,
        "adaptive": _coalesce_bool(history.get("final_adaptive"), default=True),
        "universal_idx": history.get("universal_idx"),
        "additive_residual": _coalesce_bool(
            history.get("additive_residual"),
            checkpoint_args.get("additive_residual"),
            default=False,
        ),
        "shared_lora_a": _coalesce_bool(
            history.get("shared_lora_a"),
            checkpoint_args.get("shared_lora_a"),
            default=False,
        ),
        "uemd_logit_coeff": _coalesce_float(
            history.get("uemd_logit_coeff"),
            checkpoint_args.get("uemd_logit_coeff"),
            default=0.0,
        ),
        "visa_coeff": _coalesce_float(
            history.get("visa_coeff"),
            checkpoint_args.get("visa_coeff"),
            default=0.0,
        ),
        "fedrod_dual_head": _coalesce_bool(
            history.get("fedrod_dual_head"),
            checkpoint_args.get("fedrod_dual_head"),
            default=False,
        ),
        "fedrod_alpha_init": _coalesce_float(
            history.get("fedrod_alpha_init"),
            checkpoint_args.get("fedrod_alpha_init"),
            default=2.0,
        ),
        "fedrod_universal_coeff": _coalesce_float(
            history.get("fedrod_universal_coeff"),
            checkpoint_args.get("fedrod_universal_coeff"),
            default=1.0,
        ),
        "fedrod_alpha_coeff": _coalesce_float(
            history.get("fedrod_alpha_coeff"),
            checkpoint_args.get("fedrod_alpha_coeff"),
            default=0.0,
        ),
        "cache_path": str(run_dir),
        "checkpoint_args": checkpoint_args,
    }


def _prepare_task_datasets(task_names, tokenizer, limit_tasks=None, limit_samples=None):
    selected_tasks = task_names[:limit_tasks] if limit_tasks is not None else task_names
    task_eval_datasets, task_metadata = load_full_task_validation_datasets(
        selected_tasks,
        tokenizer,
    )
    if limit_samples is not None:
        for task_name, dataset in list(task_eval_datasets.items()):
            task_eval_datasets[task_name] = dataset.select(
                range(min(limit_samples, len(dataset)))
            )
    return selected_tasks, task_eval_datasets, task_metadata


def _client_params(final_params, client_id):
    try:
        payload = final_params[client_id]
    except (KeyError, TypeError):
        payload = final_params[str(client_id)]
    return payload["params"] if isinstance(payload, dict) and "params" in payload else payload


def _clone_state(state):
    return {
        name: value.detach().clone()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def _extract_classifier_state_from_params(params, cluster_id):
    head = {
        name: value.detach().clone()
        for name, value in params.items()
        if "classifier" in name
    }
    if not head:
        raise ValueError(f"Cluster {cluster_id} representative has no classifier params")
    if any(name.startswith("cls_universal.") or ".cls_universal." in name for name in params):
        raise NotImplementedError(
            "FedRoD dual-head checkpoints contain cls_universal params; head_swap_eval.py "
            "currently supports non-FedRoD classifier heads only."
        )
    return head


def _extract_classifier_state_from_model(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if "classifier" in name
    }


def _load_classifier_state(model, head_state, *, context):
    current_state = model.state_dict()
    compatible = {}
    mismatches = []
    for name, value in head_state.items():
        if name not in current_state:
            mismatches.append(f"{name} missing")
            continue
        if tuple(current_state[name].shape) != tuple(value.shape):
            mismatches.append(
                f"{name} checkpoint {tuple(value.shape)} != model {tuple(current_state[name].shape)}"
            )
            continue
        compatible[name] = value
    if mismatches:
        raise ValueError(f"Incompatible classifier state for {context}: {mismatches}")
    model.load_state_dict(compatible, strict=False)


def _average_classifier_states(states, cluster_id):
    if not states:
        raise ValueError(f"Cluster {cluster_id} has no classifier states to average")

    keys = sorted(states[0])
    for state_idx, state in enumerate(states[1:], start=1):
        if sorted(state) != keys:
            raise ValueError(
                f"Cluster {cluster_id} classifier key mismatch at state {state_idx}: "
                f"{sorted(state)} != {keys}"
            )

    averaged = {}
    for key in keys:
        tensors = [state[key] for state in states]
        shape = tuple(tensors[0].shape)
        dtype = tensors[0].dtype
        for tensor_idx, tensor in enumerate(tensors[1:], start=1):
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"Cluster {cluster_id} classifier shape mismatch for {key} "
                    f"at state {tensor_idx}: {tuple(tensor.shape)} != {shape}"
                )
        averaged[key] = torch.stack(
            [tensor.detach().float() for tensor in tensors],
            dim=0,
        ).mean(dim=0).to(dtype=dtype)
    return averaged


def _build_cluster_heads(final_params, lora_client_map, head_source):
    cluster_heads = {}
    for cluster_id, client_ids in sorted(lora_client_map.items()):
        if not client_ids:
            raise ValueError(f"Cluster {cluster_id} has no representative clients")
        if head_source in {"cluster_average", "knn_predicted"}:
            states = [
                _extract_classifier_state_from_params(
                    _client_params(final_params, int(client_id)),
                    cluster_id,
                )
                for client_id in client_ids
            ]
            cluster_heads[int(cluster_id)] = _average_classifier_states(states, cluster_id)
        else:
            rep_client_id = int(client_ids[0])
            rep_params = _client_params(final_params, rep_client_id)
            cluster_heads[int(cluster_id)] = _extract_classifier_state_from_params(
                rep_params,
                cluster_id,
            )
    return cluster_heads


def _build_num_label_cluster_groups(cluster_metadata):
    grouped = defaultdict(list)
    for cluster_id, meta in sorted(cluster_metadata.items()):
        num_labels = meta.get("num_labels")
        if num_labels is not None:
            grouped[int(num_labels)].append(int(cluster_id))
    return {
        num_labels: sorted(cluster_ids)
        for num_labels, cluster_ids in sorted(grouped.items())
    }


def _build_shuffled_cluster_map(cluster_metadata):
    grouped = _build_num_label_cluster_groups(cluster_metadata)
    shuffled = {}
    singleton_fallback = []
    for cluster_ids in grouped.values():
        if len(cluster_ids) == 1:
            shuffled[cluster_ids[0]] = cluster_ids[0]
            singleton_fallback.append(cluster_ids[0])
            continue
        for idx, cluster_id in enumerate(cluster_ids):
            shuffled[cluster_id] = cluster_ids[(idx + 1) % len(cluster_ids)]
    return shuffled, singleton_fallback


def _cluster_prototype_task(cluster_metadata, cluster_id):
    tasks = list(cluster_metadata[int(cluster_id)].get("tasks") or [])
    if len(tasks) != 1:
        raise ValueError(
            f"KNN prototypes require one task per cluster; cluster {cluster_id} has tasks={tasks}"
        )
    return tasks[0]


def _prototype_cache_metadata(config, cluster_metadata, prototype_samples):
    return {
        "model_name": config["model_name"],
        "prototype_samples": int(prototype_samples),
        "cluster_tasks": {
            str(cluster_id): _cluster_prototype_task(cluster_metadata, cluster_id)
            for cluster_id in sorted(cluster_metadata)
        },
    }


def _cache_matches(payload, expected_metadata):
    return isinstance(payload, dict) and payload.get("metadata") == expected_metadata


def _base_cls_features(base_model, batch, device):
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key != "labels" and isinstance(value, torch.Tensor)
    }
    outputs = base_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()


def _mean_cls_prototype(base_model, dataset, tokenizer, batch_size, device, sample_count):
    sample_count = min(int(sample_count), len(dataset))
    if sample_count <= 0:
        raise ValueError("Cannot build a KNN prototype from an empty dataset")

    subset = dataset.select(range(sample_count))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    feature_sum = None
    num_seen = 0
    with torch.inference_mode():
        for batch in loader:
            features = _base_cls_features(base_model, batch, device).float()
            feature_sum = (
                features.sum(dim=0)
                if feature_sum is None
                else feature_sum + features.sum(dim=0)
            )
            num_seen += features.shape[0]
    return (feature_sum / max(num_seen, 1)).cpu()


def _predict_knn_clusters(
    base_model,
    dataset,
    tokenizer,
    prototypes,
    candidate_clusters,
    batch_size,
    device,
):
    if not candidate_clusters:
        raise ValueError("KNN prediction requires at least one candidate cluster")

    ordered_clusters = [int(cluster_id) for cluster_id in candidate_clusters]
    prototype_matrix = torch.stack(
        [prototypes[cluster_id].float() for cluster_id in ordered_clusters],
        dim=0,
    ).to(device)
    prototype_matrix = F.normalize(prototype_matrix, p=2, dim=-1)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            features = _base_cls_features(base_model, batch, device).float()
            features = F.normalize(features, p=2, dim=-1)
            nearest = torch.matmul(features, prototype_matrix.T).argmax(dim=-1).cpu()
            predictions.extend(ordered_clusters[int(index)] for index in nearest)
    return torch.tensor(predictions, dtype=torch.long)


def _load_or_build_knn_predictions(
    run_dir,
    config,
    cluster_metadata,
    task_names,
    task_eval_datasets,
    task_metadata,
    tokenizer,
    prototype_samples,
    batch_size,
):
    cache_path = run_dir / "head_swap_eval__prototypes.pt"
    metadata = _prototype_cache_metadata(config, cluster_metadata, prototype_samples)
    prototypes = None
    cache_loaded = False

    if cache_path.exists():
        payload = _torch_load_cpu(cache_path)
        if _cache_matches(payload, metadata):
            prototypes = {
                int(cluster_id): tensor.detach().cpu()
                for cluster_id, tensor in payload["prototypes"].items()
            }
            cache_loaded = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = None
    try:
        if prototypes is None:
            print(
                f"Building frozen-base KNN prototypes with {prototype_samples} examples/cluster..."
            )
            base_model = AutoModel.from_pretrained(
                config["model_name"],
                cache_dir=config["cache_path"],
            ).to(device)
            base_model.eval()

            prototypes = {}
            for cluster_id in tqdm(sorted(cluster_metadata), desc="KNN prototypes"):
                task_name = _cluster_prototype_task(cluster_metadata, cluster_id)
                if task_name not in task_eval_datasets:
                    raise ValueError(
                        f"Prototype task {task_name} for cluster {cluster_id} is not loaded"
                    )
                prototypes[int(cluster_id)] = _mean_cls_prototype(
                    base_model=base_model,
                    dataset=task_eval_datasets[task_name],
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    device=device,
                    sample_count=prototype_samples,
                )

            torch.save(
                {
                    "metadata": metadata,
                    "prototypes": {
                        int(cluster_id): tensor.cpu()
                        for cluster_id, tensor in prototypes.items()
                    },
                },
                cache_path,
            )
        else:
            print(f"Loaded frozen-base KNN prototypes from {cache_path}")

        if base_model is None:
            base_model = AutoModel.from_pretrained(
                config["model_name"],
                cache_dir=config["cache_path"],
            ).to(device)
            base_model.eval()

        clusters_by_num_labels = _build_num_label_cluster_groups(cluster_metadata)
        task_predictions = {}
        accuracy_by_task = {}
        total_correct = 0
        total_seen = 0
        for task_name in tqdm(task_names, desc="KNN target-cluster prediction"):
            target_num_labels = int(task_metadata[task_name]["num_labels"])
            candidate_clusters = clusters_by_num_labels.get(target_num_labels, [])
            predictions = _predict_knn_clusters(
                base_model=base_model,
                dataset=task_eval_datasets[task_name],
                tokenizer=tokenizer,
                prototypes=prototypes,
                candidate_clusters=candidate_clusters,
                batch_size=batch_size,
                device=device,
            )
            task_predictions[task_name] = predictions
            oracle_clusters = [
                int(cluster_id)
                for cluster_id, meta in cluster_metadata.items()
                if task_name in set(meta.get("tasks") or [])
            ]
            if len(oracle_clusters) != 1:
                raise ValueError(
                    f"Task {task_name} must map to exactly one oracle cluster, got {oracle_clusters}"
                )
            oracle_cluster = oracle_clusters[0]
            correct = int((predictions == oracle_cluster).sum().item())
            count = int(predictions.numel())
            total_correct += correct
            total_seen += count
            accuracy_by_task[task_name] = {
                "oracle_cluster": int(oracle_cluster),
                "num_examples": count,
                "correct": correct,
                "accuracy": round(100.0 * correct / count, 4) if count else math.nan,
            }

        return {
            "task_predictions": task_predictions,
            "accuracy": {
                "overall": round(100.0 * total_correct / total_seen, 4)
                if total_seen
                else math.nan,
                "correct": total_correct,
                "num_examples": total_seen,
                "by_task": accuracy_by_task,
            },
            "prototype_cache_path": str(cache_path),
            "prototype_cache_loaded": cache_loaded,
        }
    finally:
        if base_model is not None:
            del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _group_indices_by_predicted_cluster(predictions):
    grouped = defaultdict(list)
    for index, cluster_id in enumerate(predictions.tolist()):
        grouped[int(cluster_id)].append(int(index))
    return {cluster_id: indices for cluster_id, indices in sorted(grouped.items())}


def _evaluate_knn_head_dataset(
    client,
    dataset,
    predictions,
    num_labels,
    cluster_heads,
    target_cluster,
    force_routing,
    dataset_name,
):
    if len(dataset) != int(predictions.numel()):
        raise ValueError(
            f"KNN prediction length mismatch for {dataset_name}: "
            f"{predictions.numel()} predictions vs {len(dataset)} examples"
        )

    home_head = _extract_classifier_state_from_model(client.local_model)
    weighted_correct = 0.0
    total_seen = 0
    try:
        for head_cluster, indices in _group_indices_by_predicted_cluster(predictions).items():
            if not indices:
                continue
            subset = dataset.select(indices)
            _load_classifier_state(
                client.local_model,
                cluster_heads[head_cluster],
                context=(
                    f"client {client.client_id}, {dataset_name}, "
                    f"knn_head_cluster {head_cluster}, target_cluster {target_cluster}"
                ),
            )
            with _routing_override_context(
                client.local_model,
                force_routing,
                target_cluster,
            ):
                metrics = client.evaluate_on_dataset(
                    subset,
                    num_labels=num_labels,
                    dataset_name=f"{dataset_name}_knn_head_{head_cluster}",
                )
            weighted_correct += (_accuracy_from_metrics(metrics) / 100.0) * len(subset)
            total_seen += len(subset)
    finally:
        _load_classifier_state(
            client.local_model,
            home_head,
            context=f"restore client {client.client_id} home head after KNN",
        )

    return 100.0 * weighted_correct / total_seen if total_seen else math.nan


def _build_task_cluster_map(lora_client_map, task_info):
    task_to_clusters = defaultdict(set)
    for cluster_id, client_ids in lora_client_map.items():
        for client_id in client_ids:
            task_to_clusters[task_info[int(client_id)]["task_name"]].add(int(cluster_id))

    task_to_cluster = {}
    for task_name, cluster_ids in sorted(task_to_clusters.items()):
        if len(cluster_ids) != 1:
            raise ValueError(
                f"Task {task_name} appears in multiple clusters: {sorted(cluster_ids)}"
            )
        task_to_cluster[task_name] = next(iter(cluster_ids))
    return task_to_cluster


def _build_client(
    client_id,
    task_info,
    tokenizer,
    config,
    num_clients,
    client_to_cluster,
):
    client = Client(
        client_id=client_id,
        task_name=task_info[client_id]["task_name"],
        tokenizer=tokenizer,
        model_name=config["model_name"],
        num_clients=num_clients,
        rank=config["rank"],
        lora_n=config["lora_n"],
        adaptive=config["adaptive"],
        cache_path=config["cache_path"],
        idx=client_to_cluster[client_id],
        universal_idx=config["universal_idx"],
        additive_residual=config["additive_residual"],
        shared_lora_a=config["shared_lora_a"],
        uemd_logit_coeff=config["uemd_logit_coeff"],
        visa_coeff=config["visa_coeff"],
        fedrod_dual_head=config["fedrod_dual_head"],
        fedrod_alpha_init=config["fedrod_alpha_init"],
        fedrod_universal_coeff=config["fedrod_universal_coeff"],
        fedrod_alpha_coeff=config["fedrod_alpha_coeff"],
    )
    client.set_dataset(None, task_info[client_id]["num_labels"])
    return client


def _free_client_model(client):
    if client.local_model is not None:
        client._clear_routing_hooks()
        del client.local_model
        client.local_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _apply_eval_mode_override(client, force_eval_mode):
    if force_eval_mode == "auto":
        return

    def forced_eval_forward_context():
        if client.local_model is None or force_eval_mode == "natural":
            return nullcontext()
        return uemd_forward_mode(client.local_model, force_eval_mode)

    client._eval_forward_context = forced_eval_forward_context


@contextmanager
def _forced_cluster_routing(model, cluster_idx):
    sentinel = object()
    previous_indices = []
    hooks = []
    has_universal_modules = False

    for module in model.modules():
        if hasattr(module, "idx") and hasattr(module, "lora_num"):
            previous_indices.append((module, getattr(module, "idx", sentinel)))
            module.idx = int(cluster_idx)
            if getattr(module, "universal_idx", None) is not None:
                has_universal_modules = True

    def force_route(_module, _inputs, output):
        forced_logits = torch.full_like(output, torch.finfo(output.dtype).min / 2)
        forced_logits[..., int(cluster_idx)] = 0.0
        return forced_logits

    mode_context = (
        uemd_forward_mode(model, "home_cluster_only")
        if has_universal_modules
        else nullcontext()
    )
    if not has_universal_modules:
        for name, module in model.named_modules():
            if name.endswith("lora_route"):
                hooks.append(module.register_forward_hook(force_route))

    try:
        with mode_context:
            yield
    finally:
        for handle in hooks:
            handle.remove()
        for module, previous_idx in previous_indices:
            if previous_idx is sentinel:
                if hasattr(module, "idx"):
                    delattr(module, "idx")
            else:
                module.idx = previous_idx


def _routing_override_context(model, force_routing, target_cluster):
    if force_routing == "auto":
        return nullcontext()
    if force_routing == "target_cluster":
        return _forced_cluster_routing(model, target_cluster)
    raise ValueError(f"Unsupported force_routing={force_routing!r}")


def _accuracy_from_metrics(metrics):
    return metrics.get("eval_accuracy", metrics.get("accuracy", 0.0)) * 100.0


def _build_metrics_from_per_client(per_client_matrix, task_info, client_to_cluster, cluster_metadata):
    cluster_scores = defaultdict(lambda: defaultdict(list))
    for client_id_raw, task_scores in per_client_matrix.items():
        client_id = int(client_id_raw)
        cluster_id = client_to_cluster[client_id]
        cluster_label = cluster_metadata[cluster_id]["label"]
        for target_task, score in task_scores.items():
            if score is not None:
                cluster_scores[cluster_label][target_task].append(float(score))

    cluster_task_matrix = {
        cluster_label: {
            target_task: round(sum(values) / len(values), 4)
            for target_task, values in sorted(task_scores.items())
        }
        for cluster_label, task_scores in sorted(cluster_scores.items())
    }

    label_to_tasks = {
        meta["label"]: set(meta["tasks"])
        for meta in cluster_metadata.values()
    }
    diag_values = []
    offdiag_values = []
    for cluster_label, row in cluster_task_matrix.items():
        home_tasks = label_to_tasks.get(cluster_label, set())
        for target_task, score in row.items():
            if target_task in home_tasks:
                diag_values.append(float(score))
            else:
                offdiag_values.append(float(score))

    in_dist = sum(diag_values) / len(diag_values) if diag_values else math.nan
    offdiag = sum(offdiag_values) / len(offdiag_values) if offdiag_values else math.nan
    return {
        "in_dist": in_dist,
        "offdiag": offdiag,
        "matrix": cluster_task_matrix,
        "num_in_dist_cells": len(diag_values),
        "num_offdiag_cells": len(offdiag_values),
    }


def _load_existing_cross_metrics(run_dir: Path):
    path = run_dir / "cross_eval_results.json"
    if not path.exists():
        return None
    with path.open() as f:
        payload = json.load(f)

    matrix = payload.get("cluster_task_matrix") or {}
    metadata = payload.get("cluster_metadata") or {}
    diag_values = []
    offdiag_values = []
    for cluster_label, row in matrix.items():
        home_tasks = set((metadata.get(cluster_label) or {}).get("tasks") or [])
        for target_task, score in row.items():
            if score is None:
                continue
            if target_task in home_tasks:
                diag_values.append(float(score))
            else:
                offdiag_values.append(float(score))
    return {
        "path": str(path),
        "in_dist": sum(diag_values) / len(diag_values) if diag_values else math.nan,
        "offdiag": sum(offdiag_values) / len(offdiag_values) if offdiag_values else math.nan,
    }


def _evaluate_head_protocols(
    final_params,
    task_info,
    lora_client_map,
    client_to_cluster,
    cluster_metadata,
    cluster_heads,
    task_to_cluster,
    task_names,
    task_eval_datasets,
    task_metadata,
    tokenizer,
    config,
    selected_clients,
    force_eval_mode,
    head_source,
    shuffled_cluster_map,
    force_routing,
    knn_predictions=None,
):
    home_matrix = {}
    target_matrix = {}
    knn_matrix = {} if knn_predictions is not None else None

    for client_id in tqdm(selected_clients, desc="head-swap source clients"):
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
            Path(config["cache_path"])
            / f"head_swap_eval__{force_eval_mode}_{head_source}_tmp"
        )

        home_matrix[str(client_id)] = {}
        target_matrix[str(client_id)] = {}
        if knn_matrix is not None:
            knn_matrix[str(client_id)] = {}

        try:
            for target_task in task_names:
                target_num_labels = int(task_metadata[target_task]["num_labels"])
                if source_num_labels != target_num_labels:
                    home_matrix[str(client_id)][target_task] = None
                    target_matrix[str(client_id)][target_task] = None
                    if knn_matrix is not None:
                        knn_matrix[str(client_id)][target_task] = None
                    continue

                target_cluster = task_to_cluster[target_task]
                with _routing_override_context(
                    client.local_model,
                    force_routing,
                    target_cluster,
                ):
                    metrics = client.evaluate_on_dataset(
                        task_eval_datasets[target_task],
                        num_labels=target_num_labels,
                        dataset_name=f"{target_task}_validation",
                    )
                home_score = round(_accuracy_from_metrics(metrics), 4)
                home_matrix[str(client_id)][target_task] = home_score

                home_head = _extract_classifier_state_from_model(client.local_model)
                head_cluster = (
                    shuffled_cluster_map[target_cluster]
                    if head_source == "shuffled_other_cluster"
                    else target_cluster
                )
                try:
                    _load_classifier_state(
                        client.local_model,
                        cluster_heads[head_cluster],
                        context=(
                            f"client {client_id}, target_task {target_task}, "
                            f"target_cluster {target_cluster}, head_cluster {head_cluster}"
                        ),
                    )
                    with _routing_override_context(
                        client.local_model,
                        force_routing,
                        target_cluster,
                    ):
                        metrics = client.evaluate_on_dataset(
                            task_eval_datasets[target_task],
                            num_labels=target_num_labels,
                            dataset_name=f"{target_task}_validation_target_head",
                        )
                    target_score = round(_accuracy_from_metrics(metrics), 4)
                    target_matrix[str(client_id)][target_task] = target_score
                finally:
                    _load_classifier_state(
                        client.local_model,
                        home_head,
                        context=f"restore client {client_id} home head",
                    )

                if knn_matrix is not None:
                    knn_score = _evaluate_knn_head_dataset(
                        client=client,
                        dataset=task_eval_datasets[target_task],
                        predictions=knn_predictions[target_task],
                        num_labels=target_num_labels,
                        cluster_heads=cluster_heads,
                        target_cluster=target_cluster,
                        force_routing=force_routing,
                        dataset_name=f"{target_task}_validation_knn_predicted",
                    )
                    knn_matrix[str(client_id)][target_task] = round(knn_score, 4)
        finally:
            _free_client_model(client)

    home_metrics = _build_metrics_from_per_client(
        home_matrix,
        task_info,
        client_to_cluster,
        cluster_metadata,
    )
    target_metrics = _build_metrics_from_per_client(
        target_matrix,
        task_info,
        client_to_cluster,
        cluster_metadata,
    )
    knn_metrics = None
    if knn_matrix is not None:
        knn_metrics = _build_metrics_from_per_client(
            knn_matrix,
            task_info,
            client_to_cluster,
            cluster_metadata,
        )
    return home_metrics, target_metrics, knn_metrics, home_matrix, target_matrix, knn_matrix


def _round_or_nan(value):
    if math.isnan(value):
        return value
    return round(value, 4)


def _print_summary(home_metrics, target_metrics, existing_cross, knn_metrics=None):
    print("\nHead-swap summary:")
    print(
        f"  home_head:   in={home_metrics['in_dist']:.4f} "
        f"off={home_metrics['offdiag']:.4f}"
    )
    print(
        f"  target_head: in={target_metrics['in_dist']:.4f} "
        f"off={target_metrics['offdiag']:.4f}"
    )
    print(
        f"  delta:       in={target_metrics['in_dist'] - home_metrics['in_dist']:+.4f} "
        f"off={target_metrics['offdiag'] - home_metrics['offdiag']:+.4f}"
    )
    if knn_metrics is not None:
        print(
            f"  knn_head:    in={knn_metrics['in_dist']:.4f} "
            f"off={knn_metrics['offdiag']:.4f}"
        )
        print(
            f"  knn delta:   in={knn_metrics['in_dist'] - home_metrics['in_dist']:+.4f} "
            f"off={knn_metrics['offdiag'] - home_metrics['offdiag']:+.4f}"
        )
    if existing_cross is not None:
        print(
            f"  existing cross_eval: in={existing_cross['in_dist']:.4f} "
            f"off={existing_cross['offdiag']:.4f}"
        )


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
            "head_swap_eval.py currently supports non-FedRoD checkpoints only; "
            "FedRoD requires swapping cls_universal consistently too."
        )

    selected_clients = sorted(task_info)
    if args.limit_clients is not None:
        selected_clients = selected_clients[: args.limit_clients]

    print(f"Run directory: {run_dir}")
    print(f"Params source: {params_source}")
    print(f"History: {history_path}")
    print(f"Model: {config['model_name']}")
    print(f"LoRA experts: {config['lora_n']} (universal_idx={config['universal_idx']})")
    print(f"Eval context: visa_coeff={config['visa_coeff']}, uemd_logit_coeff={config['uemd_logit_coeff']}")
    print(f"Force eval mode: {args.force_eval_mode}")
    print(f"Head source: {args.head_source}")
    print(f"Force routing: {args.force_routing}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        cache_dir=config["cache_path"],
    )
    task_names = _task_order(task_info)
    task_names, task_eval_datasets, task_metadata = _prepare_task_datasets(
        task_names,
        tokenizer,
        limit_tasks=args.limit_tasks,
        limit_samples=args.limit_samples,
    )

    knn_payload = None
    if args.head_source == "knn_predicted":
        knn_payload = _load_or_build_knn_predictions(
            run_dir=run_dir,
            config=config,
            cluster_metadata=cluster_metadata,
            task_names=task_names,
            task_eval_datasets=task_eval_datasets,
            task_metadata=task_metadata,
            tokenizer=tokenizer,
            prototype_samples=args.prototype_samples,
            batch_size=args.knn_batch_size,
        )
        print(
            "KNN target-cluster accuracy: "
            f"{knn_payload['accuracy']['overall']:.4f}% "
            f"({knn_payload['accuracy']['correct']}/{knn_payload['accuracy']['num_examples']})"
        )

    cluster_heads = _build_cluster_heads(final_params, lora_client_map, args.head_source)
    task_to_cluster = _build_task_cluster_map(lora_client_map, task_info)
    shuffled_cluster_map, singleton_shuffle_fallback = _build_shuffled_cluster_map(
        cluster_metadata
    )

    (
        home_metrics,
        target_metrics,
        knn_metrics,
        home_matrix,
        target_matrix,
        knn_matrix,
    ) = _evaluate_head_protocols(
        final_params=final_params,
        task_info=task_info,
        lora_client_map=lora_client_map,
        client_to_cluster=client_to_cluster,
        cluster_metadata=cluster_metadata,
        cluster_heads=cluster_heads,
        task_to_cluster=task_to_cluster,
        task_names=task_names,
        task_eval_datasets=task_eval_datasets,
        task_metadata=task_metadata,
        tokenizer=tokenizer,
        config=config,
        selected_clients=selected_clients,
        force_eval_mode=args.force_eval_mode,
        head_source=args.head_source,
        shuffled_cluster_map=shuffled_cluster_map,
        force_routing=args.force_routing,
        knn_predictions=(knn_payload or {}).get("task_predictions"),
    )

    existing_cross = _load_existing_cross_metrics(run_dir)
    _print_summary(home_metrics, target_metrics, existing_cross, knn_metrics=knn_metrics)

    results = {
        "checkpoint_dir": str(run_dir),
        "params_source": params_source,
        "history_path": str(history_path),
        "task_order": task_names,
        "cluster_metadata": {
            str(cluster_id): meta for cluster_id, meta in cluster_metadata.items()
        },
        "task_to_cluster": {
            task_name: int(cluster_id)
            for task_name, cluster_id in sorted(task_to_cluster.items())
        },
        "config": {
            key: value
            for key, value in config.items()
            if key != "checkpoint_args"
        },
        "force_eval_mode": args.force_eval_mode,
        "head_source": args.head_source,
        "forced_routing": args.force_routing,
        "prototype_samples": int(args.prototype_samples),
        "knn_batch_size": int(args.knn_batch_size),
        "shuffled_cluster_map": {
            str(cluster_id): int(shuffled_cluster)
            for cluster_id, shuffled_cluster in sorted(shuffled_cluster_map.items())
        },
        "singleton_shuffle_fallback": [
            int(cluster_id) for cluster_id in singleton_shuffle_fallback
        ],
        "home_head_protocol": {
            "in_dist": _round_or_nan(home_metrics["in_dist"]),
            "offdiag": _round_or_nan(home_metrics["offdiag"]),
            "matrix": home_metrics["matrix"],
            "num_in_dist_cells": home_metrics["num_in_dist_cells"],
            "num_offdiag_cells": home_metrics["num_offdiag_cells"],
            "per_client_matrix": home_matrix,
        },
        "target_head_protocol": {
            "in_dist": _round_or_nan(target_metrics["in_dist"]),
            "offdiag": _round_or_nan(target_metrics["offdiag"]),
            "matrix": target_metrics["matrix"],
            "num_in_dist_cells": target_metrics["num_in_dist_cells"],
            "num_offdiag_cells": target_metrics["num_offdiag_cells"],
            "per_client_matrix": target_matrix,
        },
        "delta_off": _round_or_nan(target_metrics["offdiag"] - home_metrics["offdiag"]),
        "delta_in": _round_or_nan(target_metrics["in_dist"] - home_metrics["in_dist"]),
        "existing_cross_eval": existing_cross,
    }

    if knn_metrics is not None:
        results["predicted_routing_accuracy"] = knn_payload["accuracy"]
        results["prototype_cache_path"] = knn_payload["prototype_cache_path"]
        results["prototype_cache_loaded"] = bool(knn_payload["prototype_cache_loaded"])
        results["target_head_protocol_knn"] = {
            "in_dist": _round_or_nan(knn_metrics["in_dist"]),
            "offdiag": _round_or_nan(knn_metrics["offdiag"]),
            "matrix": knn_metrics["matrix"],
            "num_in_dist_cells": knn_metrics["num_in_dist_cells"],
            "num_offdiag_cells": knn_metrics["num_offdiag_cells"],
            "per_client_matrix": knn_matrix,
        }
        results["delta_off_knn"] = _round_or_nan(
            knn_metrics["offdiag"] - home_metrics["offdiag"]
        )
        results["delta_in_knn"] = _round_or_nan(
            knn_metrics["in_dist"] - home_metrics["in_dist"]
        )

    output_path = Path(args.output) if args.output else run_dir / "head_swap_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
