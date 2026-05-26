#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import datetime
import signal
import sys
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

# MLflow integration (optional, used by SlurmLab)
try:
    import mlflow
    MLFLOW_AVAILABLE = os.environ.get("MLFLOW_RUN_ID") is not None
except ImportError:
    MLFLOW_AVAILABLE = False

import random as _random

try:
    import submitit
except ImportError:
    submitit = None

CHECKPOINT_VERSION = 1
CHECKPOINT_DIRNAME = "checkpoints"
CHECKPOINT_POINTER_NAME = "latest.json"
MID_ROUND_SAVE_INTERVAL = 3

partition_multi_task_dataset = None
compute_lora_client_map = None
load_full_task_validation_datasets = None
compute_learned_affinity = None
build_uniform_nonhome_affinity = None
gather_cluster_signatures = None
_get_glue_task_family_impl = None
Client = None
WarmupClient = None
Server = None

ALPHA_MODE_CHOICES = ("static", "learned", "uniform_nonhome", "shuffled")
AFFINITY_MODE_DERIVATION = {
    "off": ("static", "static"),
    "train_only": ("learned", "static"),
    "full": ("learned", "learned"),
    "shuffled": ("shuffled", "shuffled"),
}


def get_glue_task_family(task_name, family_mode="oracle"):
    _ensure_glue_task_family_import()
    return _get_glue_task_family_impl(task_name, family_mode=family_mode)


def _ensure_project_imports():
    global partition_multi_task_dataset
    global compute_lora_client_map
    global load_full_task_validation_datasets
    global compute_learned_affinity
    global build_uniform_nonhome_affinity
    global gather_cluster_signatures
    global _get_glue_task_family_impl
    global get_glue_task_family
    global Client
    global WarmupClient
    global Server

    if partition_multi_task_dataset is None:
        from utils import (
            partition_multi_task_dataset as _partition_multi_task_dataset,
            compute_lora_client_map as _compute_lora_client_map,
            load_full_task_validation_datasets as _load_full_task_validation_datasets,
            compute_learned_affinity as _compute_learned_affinity,
            build_uniform_nonhome_affinity as _build_uniform_nonhome_affinity,
            get_glue_task_family as _get_glue_task_family,
        )
        from client import Client as _Client, WarmupClient as _WarmupClient
        from server import Server as _Server, gather_cluster_signatures as _gather_cluster_signatures

        partition_multi_task_dataset = _partition_multi_task_dataset
        compute_lora_client_map = _compute_lora_client_map
        load_full_task_validation_datasets = _load_full_task_validation_datasets
        compute_learned_affinity = _compute_learned_affinity
        build_uniform_nonhome_affinity = _build_uniform_nonhome_affinity
        gather_cluster_signatures = _gather_cluster_signatures
        _get_glue_task_family_impl = _get_glue_task_family
        get_glue_task_family = _get_glue_task_family
        Client = _Client
        WarmupClient = _WarmupClient
        Server = _Server


def _ensure_glue_task_family_import():
    global _get_glue_task_family_impl
    global get_glue_task_family

    if _get_glue_task_family_impl is None:
        from utils import get_glue_task_family as _get_glue_task_family

        _get_glue_task_family_impl = _get_glue_task_family
        get_glue_task_family = _get_glue_task_family


def _write_autoresearch_metrics(output_dir):
    results_path = os.environ.get("AUTORESEARCH_RESULTS_PATH")
    slurmlab_output_dir = os.environ.get("SLURMLAB_OUTPUT_DIR")
    if not results_path and not slurmlab_output_dir:
        return

    metrics = {}
    training_history_path = os.path.join(output_dir, "proposed_m2", "training_history.json")
    if os.path.exists(training_history_path):
        with open(training_history_path) as f:
            th = json.load(f)
        final = th.get("final_task_metrics") or {}
        for name, value in final.items():
            if name == "average":
                metrics["in_dist_avg"] = float(value)
            else:
                metrics[f"in_dist_{name}"] = float(value)

    cross_path = os.path.join(output_dir, "cross_eval_results.json")
    if os.path.exists(cross_path):
        with open(cross_path) as f:
            ce = json.load(f)
        matrix = ce.get("cluster_task_matrix") or {}
        diag_vals, off_vals = [], []
        cluster_meta = ce.get("cluster_metadata") or {}
        for cluster_label, row in matrix.items():
            cluster_tasks = set()
            if cluster_label in cluster_meta:
                cluster_tasks = set(cluster_meta[cluster_label].get("tasks") or [])
            for task_name, score in row.items():
                if task_name in cluster_tasks:
                    diag_vals.append(float(score))
                else:
                    off_vals.append(float(score))
        if diag_vals:
            metrics["diag_avg"] = sum(diag_vals) / len(diag_vals)
        if off_vals:
            metrics["offdiag_avg"] = sum(off_vals) / len(off_vals)

    payload = {"metrics": metrics}
    if results_path:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[autoresearch] wrote metrics to {results_path}: {metrics}")

    if slurmlab_output_dir:
        try:
            slurmlab_metrics_path = os.path.join(slurmlab_output_dir, "autoresearch_metrics.json")
            os.makedirs(slurmlab_output_dir, exist_ok=True)
            with open(slurmlab_metrics_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[autoresearch] wrote metrics to {slurmlab_metrics_path}: {metrics}")
        except OSError as exc:
            print(f"[autoresearch] failed to write slurmlab metrics: {exc}")


def _build_assignment(mode, task_name_list, num_clients, seed):
    """Build lora_client_map based on assignment mode. Returns (map, n_clusters) or None."""
    if mode == "learned":
        return None  # use default clustering
    elif mode == "oracle":
        # Group by task
        task_groups = {}
        for cid, task in enumerate(task_name_list):
            task_groups.setdefault(task, []).append(cid)
        lora_map = {i: clients for i, clients in enumerate(task_groups.values())}
        return lora_map, len(lora_map)
    elif mode == "random":
        _random.seed(seed)
        indices = list(range(num_clients))
        _random.shuffle(indices)
        n = 4  # match oracle group count
        lora_map = {i: sorted(indices[i*n:(i+1)*n]) for i in range(num_clients // n)}
        return lora_map, len(lora_map)
    elif mode == "single":
        return {0: list(range(num_clients))}, 1
    elif mode == "no_agg":
        return None  # use default clustering, but aggregation is identity
    else:
        raise ValueError(f"Unknown assignment mode: {mode}")


def _build_soft_membership(
    mode,
    lora_client_map,
    task_info,
    num_clients,
    epsilon,
    floor,
    family_mode="oracle",
):
    """Return per-client membership weights or None if mode='none'.

    family_mode='shuffled' uses the fixed seed=7 GLUE family permutation from
    utils.py, e.g. sst2->paraphrase, qnli->sentiment, mrpc/qqp->nli.
    family_mode='all_uniform' ignores family labels and spreads epsilon
    uniformly across all foreign clusters.
    """
    if mode == "none":
        return None
    if mode == "distance":
        raise NotImplementedError("--soft_membership distance is deferred to v2")
    if mode != "task_family":
        raise ValueError(f"Unknown soft_membership mode: {mode}")

    _ensure_glue_task_family_import()

    cluster_to_task = {}
    for cluster_id, client_ids in lora_client_map.items():
        if not client_ids:
            continue
        sample_client = client_ids[0]
        task_name = task_info[sample_client]["task_name"]
        cluster_to_task[int(cluster_id)] = task_name.lower()

    if family_mode not in {"oracle", "shuffled", "all_uniform"}:
        raise ValueError(f"Unknown soft membership family mode: {family_mode}")

    if family_mode == "all_uniform":
        cluster_to_family = {cid: "all_uniform" for cid in cluster_to_task}
    else:
        cluster_to_family = {
            cid: get_glue_task_family(task, family_mode=family_mode)
            for cid, task in cluster_to_task.items()
        }

    client_to_cluster = {}
    for cluster_id, client_ids in lora_client_map.items():
        for cid in client_ids:
            client_to_cluster[int(cid)] = int(cluster_id)

    membership = {}
    for client_id in range(num_clients):
        if client_id not in client_to_cluster:
            continue

        c = client_to_cluster[client_id]
        family_c = cluster_to_family[c]

        in_family_siblings = [
            k for k, fam in cluster_to_family.items() if fam == family_c and k != c
        ]
        out_of_family = [
            k for k, fam in cluster_to_family.items() if fam != family_c
        ]

        n_sib = len(in_family_siblings)
        n_oof = len(out_of_family)

        if n_sib == 0 and n_oof == 0:
            # K == 1: only home cluster, no spread possible
            weights = {c: 1.0}
        else:
            weights = {c: 1.0 - epsilon}
            if n_sib > 0 and n_oof > 0:
                # Standard split: (1-floor) of ε to in-family siblings, floor to out-of-family
                sib_weight = epsilon * (1.0 - floor) / n_sib
                oof_weight = epsilon * floor / n_oof
                for k in in_family_siblings:
                    weights[k] = sib_weight
                for k in out_of_family:
                    weights[k] = oof_weight
            elif n_sib > 0:
                # No out-of-family clusters: full ε mass distributes across siblings
                sib_weight = epsilon / n_sib
                for k in in_family_siblings:
                    weights[k] = sib_weight
            else:
                # Singleton family (no siblings): full ε mass distributes across out-of-family
                oof_weight = epsilon / n_oof
                for k in out_of_family:
                    weights[k] = oof_weight

        total = sum(weights.values())
        if abs(total - 1.0) > 1e-5:
            raise AssertionError(
                f"Soft membership for client {client_id} sums to {total:.6f} != 1.0"
            )

        membership[client_id] = weights

    return membership


def _summarize_routing_stats(client_stats):
    if not client_stats:
        return {}

    summary = {}
    for key in ["assigned_mass", "cross_mass", "universal_mass", "cluster_max_weight", "routing_entropy"]:
        values = [stats[key] for stats in client_stats.values() if key in stats]
        if values:
            summary[key] = sum(values) / len(values)

    # Average per-expert mass across clients
    per_expert_lists = [stats["per_expert_mass"] for stats in client_stats.values() if "per_expert_mass" in stats]
    if per_expert_lists:
        import numpy as _np
        summary["per_expert_mass"] = _np.mean(per_expert_lists, axis=0).round(6).tolist()

    summary["num_clients"] = len(client_stats)
    return summary


def _to_cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_state(item) for item in value]
    return value


def _args_to_dict(args):
    if args is None:
        return {}

    try:
        data = vars(args)
    except TypeError:
        data = {}

    if isinstance(data, dict) and data:
        return dict(data)

    raw_dict = getattr(args, "__dict__", None)
    if isinstance(raw_dict, dict) and raw_dict:
        return dict(raw_dict)

    payload = {}
    for key in dir(args):
        if key.startswith("_"):
            continue
        value = getattr(args, key)
        if callable(value):
            continue
        payload[key] = value
    return payload


def _checkpoint_dir(run_dir):
    checkpoint_dir = os.path.join(run_dir, CHECKPOINT_DIRNAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def _resolve_checkpoint_path(path):
    if not path:
        return None

    candidate = os.path.abspath(path)
    if candidate.endswith(".json") and os.path.exists(candidate):
        with open(candidate) as f:
            pointer = json.load(f)
        candidate = pointer.get("checkpoint_path") or pointer.get("path") or candidate

    candidate = os.path.abspath(candidate)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def _capture_rng_state():
    return {
        "python": _random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    if not state:
        return

    if state.get("python") is not None:
        _random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _phase_local_round_idx(round_idx, round_warmup):
    return round_idx if round_idx < round_warmup else round_idx - round_warmup


def _phase_local_to_global_round(progress, round_warmup):
    phase = progress.get("phase", "warmup")
    next_round_idx = int(progress.get("next_round_idx", 0))
    if phase == "clustered":
        return round_warmup + next_round_idx
    return next_round_idx


def _build_output_dir(args):
    task_name_list = args.tasks
    client_num = len(task_name_list)
    lr_str = f"{args.lr:.0e}".replace("+", "")
    return os.path.join(
        args.output_dir,
        f"{args.model_name.replace('/', '_')}_multi_task_federated_{client_num}_lr{lr_str}_seed{args.seed}",
    )


def _resolve_run_dir(output_dir):
    run_dir = os.environ.get("SLURMLAB_OUTPUT_DIR") or output_dir
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _build_progress(next_round_idx, phase, completed_client_ids, partial_client_params, active_client_id):
    return {
        "next_round_idx": int(next_round_idx),
        "phase": phase,
        "completed_client_ids": sorted(int(client_id) for client_id in completed_client_ids),
        "partial_client_params": {
            int(client_id): _to_cpu_state(params)
            for client_id, params in partial_client_params.items()
        },
        "active_client_id": None if active_client_id is None else int(active_client_id),
    }


def _build_fed_state(
    aggregated_params,
    saved_params,
    lora_client_map,
    optimal_n_clusters,
    universal_idx,
    universal_init_params,
    soft_membership=None,
    per_cluster_init_params=None,
    assignment_mode=None,
    soft_membership_mode="none",
    soft_membership_family_mode="oracle",
    soft_train_weight=0.0,
    soft_floor=0.20,
    uemd_logit_coeff=0.0,
    visa_coeff=0.0,
    visa_conflict_clip=False,
    fedrod_dual_head=False,
    fedrod_alpha_init=2.0,
    fedrod_universal_coeff=1.0,
    fedrod_alpha_coeff=0.0,
    rdrop_kl_coeff=0.0,
    rdrop_direction="symmetric",
    rdrop_stopgrad="target",
    affinity_mode="off",
    client_alpha_mode=None,
    server_alpha_mode=None,
    learned_affinity=None,
):
    if client_alpha_mode is None or server_alpha_mode is None:
        client_alpha_mode, server_alpha_mode = _resolve_alpha_modes(
            affinity_mode,
            client_alpha=client_alpha_mode,
            server_alpha=server_alpha_mode,
        )
    needs_alpha_state = (
        client_alpha_mode != "static" or server_alpha_mode != "static"
    )
    fed_state = {
        "aggregated_params": _to_cpu_state(aggregated_params),
        "saved_params": _to_cpu_state(saved_params),
        "lora_client_map": _to_cpu_state(lora_client_map),
        "optimal_n_clusters": optimal_n_clusters,
        "universal_idx": universal_idx,
        "universal_init_params": _to_cpu_state(universal_init_params),
        "soft_membership": _to_cpu_state(soft_membership) if soft_membership else None,
        "per_cluster_init_params": _to_cpu_state(per_cluster_init_params) if per_cluster_init_params else None,
        "assignment_mode": assignment_mode,
        "soft_membership_mode": soft_membership_mode,
        "soft_membership_family_mode": soft_membership_family_mode,
        "soft_train_weight": float(soft_train_weight),
        "soft_floor": float(soft_floor),
        "uemd_logit_coeff": float(uemd_logit_coeff),
        "visa_coeff": float(visa_coeff),
        "visa_conflict_clip": bool(visa_conflict_clip),
        "fedrod_dual_head": bool(fedrod_dual_head),
        "fedrod_alpha_init": float(fedrod_alpha_init),
        "fedrod_universal_coeff": float(fedrod_universal_coeff),
        "fedrod_alpha_coeff": float(fedrod_alpha_coeff),
        "rdrop_kl_coeff": float(rdrop_kl_coeff),
        "rdrop_direction": rdrop_direction,
        "rdrop_stopgrad": rdrop_stopgrad,
        "client_alpha_mode": client_alpha_mode,
        "server_alpha_mode": server_alpha_mode,
    }
    if needs_alpha_state:
        fed_state["learned_affinity"] = _to_cpu_state(learned_affinity)
    return fed_state


def _build_history_state(client_scores, routing_stats, task_info):
    return {
        "client_scores": _to_cpu_state(client_scores),
        "routing_stats": _to_cpu_state(routing_stats),
        "task_info": _to_cpu_state(task_info),
    }


def _write_checkpoint_pointer(checkpoint_dir, checkpoint_path, progress):
    pointer_path = os.path.join(checkpoint_dir, CHECKPOINT_POINTER_NAME)
    payload = {
        "checkpoint_path": checkpoint_path,
        "path": checkpoint_path,
        "complete": True,
        "round_idx": int(progress["next_round_idx"]),
        "phase": progress["phase"],
        "version": CHECKPOINT_VERSION,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(pointer_path, "w") as f:
        json.dump(payload, f, indent=2)
    return pointer_path


def _save_training_checkpoint(run_dir, args, tag, progress, fed_state, history):
    checkpoint_dir = _checkpoint_dir(run_dir)
    checkpoint_path = os.path.abspath(
        os.path.join(checkpoint_dir, f"round_{int(progress['next_round_idx'])}_{tag}.pt")
    )
    payload = {
        "version": CHECKPOINT_VERSION,
        "tag": tag,
        "args": _args_to_dict(args),
        "progress": _to_cpu_state(progress),
        "fed_state": _to_cpu_state(fed_state),
        "history": _to_cpu_state(history),
        "rng": _capture_rng_state(),
    }
    torch.save(payload, checkpoint_path)
    _write_checkpoint_pointer(checkpoint_dir, checkpoint_path, payload["progress"])
    return checkpoint_path


def _load_training_checkpoint(path):
    resolved_path = _resolve_checkpoint_path(path)
    try:
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved_path, map_location="cpu")

    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {checkpoint.get('version')} in {resolved_path}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    return checkpoint


def _validate_resume_args(args, resume_state):
    saved_args = resume_state.get("args") or {}
    current_args = _args_to_dict(args)

    structural_errors = []
    for key in [
        "model_name",
        "tasks",
        "max_clusters",
        "rank",
        "warmup_rounds",
        "universal_expert",
        "additive_residual",
        "soft_membership",
        "soft_membership_family_mode",
        "affinity_mode",
        "client_alpha_mode",
        "server_alpha_mode",
        "fedrod_dual_head",
    ]:
        if key in saved_args and key in current_args and saved_args[key] != current_args[key]:
            structural_errors.append(
                f"{key}: checkpoint={saved_args[key]!r}, current={current_args[key]!r}"
            )

    saved_tasks = saved_args.get("tasks")
    current_tasks = current_args.get("tasks")
    if saved_tasks is not None and current_tasks is not None:
        saved_num_clients = len(saved_tasks)
        current_num_clients = len(current_tasks)
        if saved_num_clients != current_num_clients:
            structural_errors.append(
                f"num_clients: checkpoint={saved_num_clients!r}, current={current_num_clients!r}"
            )

    if structural_errors:
        raise ValueError(
            "Refusing to resume from checkpoint with incompatible structural args:\n- "
            + "\n- ".join(structural_errors)
        )

    for key in ["lr", "seed", "batch_size"]:
        if key in saved_args and key in current_args and saved_args[key] != current_args[key]:
            print(
                f"[resume] warning: overriding checkpoint {key}={saved_args[key]!r} "
                f"with current value {current_args[key]!r}"
            )


def _initialize_clustered_clients(
    warmup_clients,
    task_info,
    client_datasets,
    lora_client_map,
    optimal_n_clusters,
    universal_expert,
    universal_idx,
    additive_residual,
    rank,
    output_dir,
    shared_lora_a=False,
    visa_coeff=0.0,
    fedrod_dual_head=False,
    fedrod_alpha_init=2.0,
    fedrod_universal_coeff=1.0,
    fedrod_alpha_coeff=0.0,
    rdrop_kl_coeff=0.0,
    rdrop_direction="symmetric",
    rdrop_stopgrad="target",
):
    _ensure_project_imports()

    if not lora_client_map or task_info is None or client_datasets is None or optimal_n_clusters is None:
        return None, 1

    clustered_clients = []
    clustered_lora_n = optimal_n_clusters + 1 if universal_expert else optimal_n_clusters
    for client_id in range(len(warmup_clients)):
        client_task = task_info[client_id]["task_name"]
        num_labels = task_info[client_id]["num_labels"]

        client_cluster = None
        for cluster_id, cluster_clients in lora_client_map.items():
            if client_id in cluster_clients:
                client_cluster = int(cluster_id)
                break

        if client_cluster is None:
            print(f"Warning: Client {client_id} not found in any cluster. Assigning to cluster 0.")
            client_cluster = 0

        client = Client(
            client_id=client_id,
            task_name=client_task,
            tokenizer=warmup_clients[client_id].tokenizer,
            model_name=warmup_clients[client_id].model_name,
            num_clients=len(warmup_clients),
            rank=rank,
            lora_n=clustered_lora_n,
            adaptive=True,
            cache_path=output_dir,
            idx=client_cluster,
            universal_idx=universal_idx,
            additive_residual=additive_residual,
            shared_lora_a=shared_lora_a,
            visa_coeff=visa_coeff,
            fedrod_dual_head=fedrod_dual_head,
            fedrod_alpha_init=fedrod_alpha_init,
            fedrod_universal_coeff=fedrod_universal_coeff,
            fedrod_alpha_coeff=fedrod_alpha_coeff,
            rdrop_kl_coeff=rdrop_kl_coeff,
            rdrop_direction=rdrop_direction,
            rdrop_stopgrad=rdrop_stopgrad,
        )
        client.set_dataset(client_datasets[client_id], num_labels)
        clustered_clients.append(client)

    return clustered_clients, clustered_lora_n


class CheckpointController:
    def __init__(self, run_dir, args, client_save_interval=MID_ROUND_SAVE_INTERVAL):
        self.run_dir = os.path.abspath(run_dir)
        self.args = args
        self.client_save_interval = max(1, int(client_save_interval))
        self._latest_snapshot = None
        self._signal_requested = False
        self._installed = False

    def install(self):
        if self._installed:
            return
        for signal_name in ("SIGTERM", "SIGUSR1", "SIGUSR2"):
            if hasattr(signal, signal_name):
                signal.signal(getattr(signal, signal_name), self._handle_signal)
        self._installed = True

    def update_snapshot(self, tag, progress, fed_state, history):
        self._latest_snapshot = {
            "tag": tag,
            "progress": _to_cpu_state(progress),
            "fed_state": _to_cpu_state(fed_state),
            "history": _to_cpu_state(history),
        }

    def maybe_save(self, tag, progress, fed_state, history, force=False):
        self.update_snapshot(tag=tag, progress=progress, fed_state=fed_state, history=history)

        is_mid_round = tag.startswith("round_in_progress_client_")
        should_save = force or self._signal_requested or not is_mid_round
        if is_mid_round and not should_save:
            completed_count = len(progress.get("completed_client_ids") or [])
            should_save = completed_count > 0 and completed_count % self.client_save_interval == 0

        if not should_save:
            return None

        checkpoint_path = _save_training_checkpoint(
            run_dir=self.run_dir,
            args=self.args,
            tag=tag,
            progress=progress,
            fed_state=fed_state,
            history=history,
        )
        self._signal_requested = False
        print(f"[checkpoint] saved {checkpoint_path}")
        return checkpoint_path

    def _handle_signal(self, signum, _frame):
        self._signal_requested = True
        signal_name = signal.Signals(signum).name
        print(f"[checkpoint] received {signal_name}")
        if self._latest_snapshot is not None:
            snapshot = self._latest_snapshot
            checkpoint_path = self.maybe_save(
                tag=snapshot["tag"],
                progress=snapshot["progress"],
                fed_state=snapshot["fed_state"],
                history=snapshot["history"],
                force=True,
            )
            if checkpoint_path is not None:
                print(f"[checkpoint] saved signal-triggered checkpoint to {checkpoint_path}")
        raise SystemExit(0)


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


def _normalize_cluster_affinity(cluster_affinity):
    if not cluster_affinity:
        return None
    return {
        int(home_cluster): {
            int(expert_idx): float(weight)
            for expert_idx, weight in weights.items()
        }
        for home_cluster, weights in cluster_affinity.items()
    }


def _normalize_cluster_signatures(cluster_signatures):
    if not cluster_signatures:
        return None
    normalized = {}
    for cluster_id, signature in cluster_signatures.items():
        if isinstance(signature, torch.Tensor):
            signature = signature.detach().float().cpu().tolist()
        else:
            signature = torch.tensor(signature, dtype=torch.float32).tolist()
        normalized[int(cluster_id)] = signature
    return normalized


def _normalize_shuffle_permutation(shuffle_permutation):
    if not shuffle_permutation:
        return None
    if isinstance(shuffle_permutation, (list, tuple)):
        return {
            int(cluster_id): int(shuffled_id)
            for cluster_id, shuffled_id in enumerate(shuffle_permutation)
        }
    return {
        int(cluster_id): int(shuffled_id)
        for cluster_id, shuffled_id in shuffle_permutation.items()
    }


def _normalize_learned_affinity_state(learned_affinity):
    if not learned_affinity:
        return None

    last_refresh_round = learned_affinity.get("last_refresh_round")
    current_alpha = _normalize_cluster_affinity(
        learned_affinity.get("current_alpha")
    )
    shuffle_permutation = _normalize_shuffle_permutation(
        learned_affinity.get("shuffle_permutation")
    )

    has_new_schema = any(
        key in learned_affinity
        for key in (
            "client_alpha",
            "server_alpha",
            "client_shuffle_permutation",
            "server_shuffle_permutation",
        )
    )
    if has_new_schema:
        client_alpha = (
            _normalize_cluster_affinity(learned_affinity.get("client_alpha"))
            if "client_alpha" in learned_affinity
            else current_alpha
        )
        server_alpha = _normalize_cluster_affinity(
            learned_affinity.get("server_alpha")
        )
        client_shuffle_permutation = (
            _normalize_shuffle_permutation(
                learned_affinity.get("client_shuffle_permutation")
            )
            if "client_shuffle_permutation" in learned_affinity
            else shuffle_permutation
        )
        server_shuffle_permutation = _normalize_shuffle_permutation(
            learned_affinity.get("server_shuffle_permutation")
        )
    else:
        # Old checkpoints only had aliases. With no saved side flags in this
        # nested state, infer shuffled from the presence of a permutation;
        # otherwise preserve the old full-style behavior on both sides.
        client_alpha = current_alpha
        server_alpha = current_alpha if current_alpha is not None else None
        client_shuffle_permutation = shuffle_permutation
        server_shuffle_permutation = (
            shuffle_permutation
            if current_alpha is not None and shuffle_permutation is not None
            else None
        )

    return {
        "last_refresh_round": (
            None if last_refresh_round is None else int(last_refresh_round)
        ),
        "current_signatures": _normalize_cluster_signatures(
            learned_affinity.get("current_signatures")
        ),
        "client_alpha": client_alpha,
        "server_alpha": server_alpha,
        "client_shuffle_permutation": client_shuffle_permutation,
        "server_shuffle_permutation": server_shuffle_permutation,
        "current_alpha": client_alpha,
        "shuffle_permutation": client_shuffle_permutation,
    }


def _expand_cluster_affinity_to_clients(
    current_alpha,
    client_to_cluster,
    num_clients,
    base_membership=None,
):
    if not current_alpha:
        return None

    learned_membership = {}
    for client_id in range(num_clients):
        home_cluster = int(client_to_cluster.get(client_id, 0))
        weights = (
            deepcopy(base_membership.get(client_id, {}))
            if base_membership and client_id in base_membership
            else {}
        )
        weights.setdefault(home_cluster, 1.0)
        for expert_idx, alpha in current_alpha.get(home_cluster, {}).items():
            weights[int(expert_idx)] = float(alpha)
        learned_membership[client_id] = weights

    return learned_membership


def _should_refresh_learned_affinity(
    round_number,
    round_warmup,
    affinity_refresh_rounds,
    last_refresh_round,
):
    if round_number < round_warmup:
        return False
    if last_refresh_round is None:
        return round_number == round_warmup
    return round_number >= int(last_refresh_round) + int(affinity_refresh_rounds)


def _collect_client_home_signature(client, batch_size, num_batches=8):
    if client.local_model is None:
        raise RuntimeError("client model must be loaded before collecting a home signature")

    collector = getattr(client, "collect_home_signature", None)
    if callable(collector):
        return collector(client.local_model, num_batches=num_batches)

    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    from client import _get_fedrod_backbone, _get_sequence_classifier_model
    from peft.tuners.lora import uemd_forward_mode

    max_batches = min(int(num_batches), 8)
    if max_batches <= 0:
        return None

    sequence_model = _get_sequence_classifier_model(client.local_model)
    backbone = _get_fedrod_backbone(sequence_model)
    train_dataset = client.datasets["train"]
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(client.tokenizer),
    )

    was_training = client.local_model.training
    client.local_model.eval()
    signature_sum = None
    example_count = 0

    try:
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                device = next(client.local_model.parameters()).device
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                signature_inputs = {
                    key: value
                    for key, value in inputs.items()
                    if key not in {"labels", "label", "label_ids"}
                }
                with uemd_forward_mode(client.local_model, "home_cluster_only"):
                    outputs = backbone(**signature_inputs, return_dict=True)

                cls_hidden = outputs.last_hidden_state[:, 0, :].detach().float().cpu()
                signature_sum = (
                    cls_hidden.sum(dim=0)
                    if signature_sum is None
                    else signature_sum + cls_hidden.sum(dim=0)
                )
                example_count += int(cls_hidden.shape[0])
    finally:
        if was_training:
            client.local_model.train()

    if signature_sum is None or example_count == 0:
        return None
    return signature_sum / example_count


def _collect_signature_from_params(client, params, batch_size):
    loaded_here = client.local_model is None
    if loaded_here:
        client.load_model()
    if params is not None:
        client.load_params(params)
    try:
        return _collect_client_home_signature(client, batch_size=batch_size, num_batches=8)
    finally:
        if loaded_here:
            client.unload_model()


def _refresh_client_visa_weights(clients, soft_membership):
    if soft_membership is None:
        return
    for client in clients:
        refresh = getattr(client, "refresh_visa_weights", None)
        if callable(refresh):
            refresh(soft_membership.get(client.client_id))


def _build_alpha_for_side(
    mode,
    cluster_signatures,
    tau,
    shuffle_permutation,
    cluster_ids,
    shuffle_seed,
):
    if mode == "static":
        return None
    if mode == "uniform_nonhome":
        return build_uniform_nonhome_affinity(cluster_ids)
    if cluster_signatures is None:
        raise RuntimeError(f"{mode} alpha requires collected cluster signatures")
    if mode == "learned":
        return compute_learned_affinity(cluster_signatures, tau=tau, mode="full")
    if mode == "shuffled":
        return compute_learned_affinity(
            cluster_signatures,
            tau=tau,
            mode="shuffled",
            shuffle_seed=shuffle_seed,
            shuffle_permutation=shuffle_permutation,
        )
    raise ValueError(f"Unknown alpha mode: {mode}")


def _sorted_shuffle_permutation(shuffle_permutation):
    if shuffle_permutation is None:
        return None
    return {
        int(cluster_id): int(shuffled_id)
        for cluster_id, shuffled_id in sorted(shuffle_permutation.items())
    }


def _existing_shuffle_permutation(learned_affinity, side):
    if not learned_affinity:
        return None
    key = f"{side}_shuffle_permutation"
    permutation = learned_affinity.get(key)
    if permutation is None and side == "client":
        permutation = learned_affinity.get("shuffle_permutation")
    return _normalize_shuffle_permutation(permutation)


def _build_side_shuffle_permutations(
    client_alpha_mode,
    server_alpha_mode,
    cluster_ids,
    affinity_shuffle_seed,
    previous_state=None,
):
    from utils import build_learned_affinity_derangement

    client_is_shuffled = client_alpha_mode == "shuffled"
    server_is_shuffled = server_alpha_mode == "shuffled"
    if client_is_shuffled and server_is_shuffled:
        shared_pi = (
            _existing_shuffle_permutation(previous_state, "client")
            or _existing_shuffle_permutation(previous_state, "server")
            or build_learned_affinity_derangement(cluster_ids, affinity_shuffle_seed)
        )
        shared_pi = _sorted_shuffle_permutation(shared_pi)
        return shared_pi, shared_pi
    if client_is_shuffled:
        client_pi = (
            _existing_shuffle_permutation(previous_state, "client")
            or build_learned_affinity_derangement(cluster_ids, affinity_shuffle_seed)
        )
        return _sorted_shuffle_permutation(client_pi), None
    if server_is_shuffled:
        server_pi = (
            _existing_shuffle_permutation(previous_state, "server")
            or build_learned_affinity_derangement(cluster_ids, affinity_shuffle_seed)
        )
        return None, _sorted_shuffle_permutation(server_pi)
    return None, None


def _refresh_learned_affinity_state(
    client_signatures,
    lora_client_map,
    num_clients,
    round_number,
    affinity_tau,
    client_alpha_mode,
    server_alpha_mode,
    affinity_shuffle_seed,
    base_membership,
    log_file,
    previous_state=None,
    log_shuffle_derangement=False,
):
    needs_dynamic_alpha = (
        client_alpha_mode in {"learned", "shuffled"}
        or server_alpha_mode in {"learned", "shuffled"}
    )
    if not lora_client_map:
        raise RuntimeError("alpha state requires lora_client_map")
    if needs_dynamic_alpha:
        if client_signatures is None:
            raise RuntimeError("dynamic alpha refresh requires client signatures")
        cluster_signatures = gather_cluster_signatures(client_signatures, lora_client_map)
        cluster_ids = sorted(int(cluster_id) for cluster_id in cluster_signatures)
    else:
        cluster_signatures = None
        cluster_ids = sorted(int(cluster_id) for cluster_id in lora_client_map)

    client_pi, server_pi = _build_side_shuffle_permutations(
        client_alpha_mode,
        server_alpha_mode,
        cluster_ids,
        affinity_shuffle_seed,
        previous_state=previous_state,
    )
    client_alpha = _build_alpha_for_side(
        client_alpha_mode,
        cluster_signatures,
        affinity_tau,
        client_pi,
        cluster_ids,
        affinity_shuffle_seed,
    )
    server_alpha = _build_alpha_for_side(
        server_alpha_mode,
        cluster_signatures,
        affinity_tau,
        server_pi,
        cluster_ids,
        affinity_shuffle_seed,
    )
    client_to_cluster = _build_client_cluster_map(lora_client_map, num_clients)
    learned_client_soft_membership = _expand_cluster_affinity_to_clients(
        client_alpha,
        client_to_cluster,
        num_clients,
        base_membership=base_membership,
    )
    learned_server_soft_membership = _expand_cluster_affinity_to_clients(
        server_alpha,
        client_to_cluster,
        num_clients,
        base_membership=base_membership,
    )
    current_signatures = (
        {
            int(cluster_id): signature.detach().float().cpu().tolist()
            for cluster_id, signature in cluster_signatures.items()
        }
        if cluster_signatures is not None
        else None
    )
    learned_affinity = {
        "current_signatures": current_signatures,
        "last_refresh_round": int(round_number),
        "client_alpha": client_alpha,
        "server_alpha": server_alpha,
        "client_shuffle_permutation": client_pi,
        "server_shuffle_permutation": server_pi,
        "current_alpha": client_alpha,
        "shuffle_permutation": client_pi,
    }

    if (
        log_shuffle_derangement
        and (client_pi is not None or server_pi is not None)
    ):
        if client_pi is not None and server_pi is client_pi:
            permutation_text = ", ".join(
                f"{source_id}->{target_id}"
                for source_id, target_id in sorted(client_pi.items())
            )
            shuffle_msg = (
                f"[learned_affinity] mode=shuffled, shared_derangement="
                f"{{{permutation_text}}}"
            )
        else:
            parts = []
            for side, permutation in (
                ("client", client_pi),
                ("server", server_pi),
            ):
                if permutation is None:
                    continue
                permutation_text = ", ".join(
                    f"{source_id}->{target_id}"
                    for source_id, target_id in sorted(permutation.items())
                )
                parts.append(f"{side}={{{permutation_text}}}")
            shuffle_msg = f"[learned_affinity] mode=shuffled, derangement {'; '.join(parts)}"
        print(shuffle_msg)
        with open(log_file, "a") as f:
            f.write(shuffle_msg + "\n")

    msg = (
        f"[learned_affinity] round={round_number} refreshed: "
        f"{len(cluster_ids)} clusters, "
        f"client_exposure_mode={client_alpha_mode}, server_exposure_mode={server_alpha_mode}"
    )
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    return learned_client_soft_membership, learned_server_soft_membership, learned_affinity


def _build_cluster_metadata(lora_client_map, task_info, num_clients):
    if not lora_client_map:
        unique_tasks = sorted({task_info[i]["task_name"] for i in range(num_clients)}) if task_info else []
        return {
            0: {
                "label": "all_clients",
                "clients": list(range(num_clients)),
                "tasks": unique_tasks,
            }
        }

    metadata = {}
    used_labels = set()
    for raw_cluster_id, cluster_clients in sorted(lora_client_map.items(), key=lambda item: int(item[0])):
        cluster_id = int(raw_cluster_id)
        cluster_clients = [int(client_id) for client_id in cluster_clients]
        cluster_tasks = sorted({task_info[client_id]["task_name"] for client_id in cluster_clients}) if task_info else []

        base_label = cluster_tasks[0] if len(cluster_tasks) == 1 else f"cluster_{cluster_id}"
        label = base_label if base_label not in used_labels else f"{base_label}_cluster_{cluster_id}"
        used_labels.add(label)

        metadata[cluster_id] = {
            "label": label,
            "clients": cluster_clients,
            "tasks": cluster_tasks,
        }

    return metadata


def _compute_universal_init_params(saved_params):
    if not saved_params:
        return {}

    reference_params = next(iter(saved_params.values()))
    universal_params = {}

    for name in reference_params:
        if "lora_A0" not in name and "lora_B0" not in name:
            continue

        stacked_params = torch.stack([
            client_params[name].detach().cpu()
            for client_params in saved_params.values()
            if name in client_params
        ])
        universal_params[name] = stacked_params.mean(dim=0)

    return universal_params


def _compute_per_cluster_init_params(saved_params, lora_client_map):
    """Average warmup lora_A0/lora_B0 weights for each cluster."""
    if not saved_params or not lora_client_map:
        return {}

    per_cluster_init = {}
    for cluster_id, client_ids in lora_client_map.items():
        cluster_id = int(cluster_id)
        cluster_clients = [
            cid for cid in client_ids
            if cid in saved_params and saved_params[cid] is not None
        ]
        if not cluster_clients:
            continue

        sample_params = saved_params[cluster_clients[0]]
        init_for_cluster = {}
        for name in sample_params:
            if "lora_A0" not in name and "lora_B0" not in name:
                continue
            stacked = torch.stack([
                saved_params[cid][name].detach().cpu()
                for cid in cluster_clients
                if name in saved_params[cid]
            ])
            init_for_cluster[name] = stacked.mean(dim=0)

        if init_for_cluster:
            per_cluster_init[cluster_id] = init_for_cluster

    return per_cluster_init


def _clone_classifier_param_to_cls_universal_name(param_name):
    marker = ".classifier."
    if marker in param_name:
        return "cls_universal." + param_name.split(marker, 1)[1]
    if param_name.startswith("classifier."):
        return "cls_universal." + param_name.split("classifier.", 1)[1]
    return None


def _build_clustered_warm_start(
    client,
    client_params,
    universal_init_params=None,
    per_cluster_init_params=None,
):
    warmed_params = {}

    if per_cluster_init_params is not None:
        for name, param in client_params.items():
            if "lora_A0" in name or "lora_B0" in name or "lora_route" in name:
                continue
            warmed_params[name] = param.clone() if isinstance(param, torch.Tensor) else param

        for cluster_id, init in per_cluster_init_params.items():
            for source_name, tensor in init.items():
                if "lora_A0" in source_name:
                    target = source_name.replace("lora_A0", f"lora_A{cluster_id}")
                elif "lora_B0" in source_name:
                    target = source_name.replace("lora_B0", f"lora_B{cluster_id}")
                else:
                    continue
                warmed_params[target] = tensor.clone()
    else:
        for name, param in client_params.items():
            if "lora_A0" in name and client.idx is not None:
                warmed_params[name.replace("lora_A0", f"lora_A{client.idx}")] = param.clone()
            elif "lora_B0" in name and client.idx is not None:
                warmed_params[name.replace("lora_B0", f"lora_B{client.idx}")] = param.clone()
            elif "lora_route" in name:
                continue
            else:
                warmed_params[name] = param.clone() if isinstance(param, torch.Tensor) else param

    if getattr(client, "fedrod_dual_head", False):
        for name, param in client_params.items():
            target = _clone_classifier_param_to_cls_universal_name(name)
            if target is not None:
                warmed_params[target] = param.clone() if isinstance(param, torch.Tensor) else param

    if universal_init_params and client.universal_idx is not None:
        for name, param in universal_init_params.items():
            if "lora_A0" in name:
                warmed_params[name.replace("lora_A0", f"lora_A{client.universal_idx}")] = param.clone()
            elif "lora_B0" in name:
                warmed_params[name.replace("lora_B0", f"lora_B{client.universal_idx}")] = param.clone()

    return warmed_params


def _run_cross_task_evaluation(final_clients, aggregated_params, task_info, lora_client_map, output_dir):
    _ensure_project_imports()

    if not final_clients or not task_info:
        return None

    task_names = list(dict.fromkeys(task_info[i]["task_name"] for i in range(len(final_clients))))
    task_eval_datasets, task_metadata = load_full_task_validation_datasets(
        task_names,
        final_clients[0].tokenizer,
    )

    client_to_cluster = _build_client_cluster_map(lora_client_map, len(final_clients))
    cluster_metadata = _build_cluster_metadata(lora_client_map, task_info, len(final_clients))

    per_client_matrix = {}
    cluster_scores = defaultdict(lambda: defaultdict(list))

    for client in final_clients:
        client_id = client.client_id
        client.load_model()
        client.load_params(aggregated_params[client_id])

        task_scores = {}
        for target_task in task_names:
            target_num_labels = task_metadata[target_task]["num_labels"]
            if client.num_labels != target_num_labels:
                task_scores[target_task] = None
                continue
            metrics = client.evaluate_on_dataset(
                task_eval_datasets[target_task],
                num_labels=target_num_labels,
                dataset_name=f"{target_task}_validation",
            )
            accuracy = metrics.get("eval_accuracy", metrics.get("accuracy", 0.0)) * 100
            task_scores[target_task] = round(accuracy, 4)

            cluster_id = client_to_cluster.get(client_id, 0)
            cluster_label = cluster_metadata[cluster_id]["label"]
            cluster_scores[cluster_label][target_task].append(task_scores[target_task])

        per_client_matrix[str(client_id)] = task_scores
        client.unload_model()

    cluster_task_matrix = {
        cluster_label: {
            task_name: round(sum(values) / len(values), 4)
            for task_name, values in sorted(task_scores.items())
        }
        for cluster_label, task_scores in sorted(cluster_scores.items())
    }

    serialized_cluster_metadata = {
        meta["label"]: {
            "cluster_id": cluster_id,
            "clients": meta["clients"],
            "tasks": meta["tasks"],
        }
        for cluster_id, meta in sorted(cluster_metadata.items())
    }

    results = {
        "cluster_task_matrix": cluster_task_matrix,
        "cluster_metadata": serialized_cluster_metadata,
        "per_client_matrix": per_client_matrix,
        "task_order": task_names,
    }

    with open(os.path.join(output_dir, "cross_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def train_federated(
    dummy,
    clients,
    server,
    global_rounds,
    local_epochs,
    output_dir,
    lr=3e-4,
    round_warmup=1,
    max_clusters=10,
    assignment_mode="learned",
    adaptive_delay=0,
    task_info=None,
    client_datasets=None,
    batch_size=128,
    rank=4,
    save_final_params=True,
    cross_eval=False,
    universal_expert=False,
    universal_warmup_rounds=2,
    load_balance_coeff=0.01,
    uemd_coeff=0.0,
    uemd_logit_coeff=0.0,
    visa_coeff=0.0,
    visa_conflict_clip=False,
    fedrod_dual_head=False,
    fedrod_alpha_init=2.0,
    fedrod_universal_coeff=1.0,
    fedrod_alpha_coeff=0.0,
    rdrop_kl_coeff=0.0,
    rdrop_direction="symmetric",
    rdrop_stopgrad="target",
    additive_residual=False,
    affinity_mode="off",
    client_alpha_mode=None,
    server_alpha_mode=None,
    affinity_tau=0.5,
    affinity_refresh_rounds=5,
    affinity_shuffle_seed=7,
    soft_membership_mode="none",
    soft_membership_family_mode="oracle",
    soft_train_weight=0.20,
    soft_floor=0.20,
    shared_lora_a=False,
    resume_state=None,
    checkpoint_controller=None,
):
    _ensure_project_imports()
    if client_alpha_mode is None or server_alpha_mode is None:
        client_alpha_mode, server_alpha_mode = _resolve_alpha_modes(
            affinity_mode,
            client_alpha=client_alpha_mode,
            server_alpha=server_alpha_mode,
        )
    alpha_predicates = _alpha_mode_predicates(client_alpha_mode, server_alpha_mode)
    client_uses_side_alpha = alpha_predicates["client_uses_side_alpha"]
    server_uses_side_alpha = alpha_predicates["server_uses_side_alpha"]
    client_uses_dynamic_alpha = alpha_predicates["client_uses_dynamic_alpha"]
    server_uses_dynamic_alpha = alpha_predicates["server_uses_dynamic_alpha"]
    needs_signatures = alpha_predicates["needs_signatures"]
    needs_alpha_state = alpha_predicates["needs_alpha_state"]

    personal_dir = os.path.join(output_dir, "proposed_m2")
    os.makedirs(personal_dir, exist_ok=True)
    log_file = os.path.join(personal_dir, "training_log.txt")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_mode = "a" if resume_state and os.path.exists(log_file) else "w"
    with open(log_file, log_mode) as f:
        if log_mode == "w":
            f.write(f"[{current_time}] Starting Federated Training with Dummy Client\n")
            f.write(
                f"Total Rounds: {global_rounds}, Local Epochs: {local_epochs}, Warmup Rounds: {round_warmup}\n"
            )
        else:
            f.write(f"\n[{current_time}] Resuming Federated Training from checkpoint\n")
        f.write("-" * 50 + "\n")

    warmup_clients = clients
    resume_progress = (resume_state or {}).get("progress") or {}
    resume_fed_state = (resume_state or {}).get("fed_state") or {}
    resume_history = (resume_state or {}).get("history") or {}
    resume_tag = (resume_state or {}).get("tag")

    all_client_scores = {
        int(client_id): scores
        for client_id, scores in (resume_history.get("client_scores") or {}).items()
    } or {client.client_id: [] for client in warmup_clients}
    for client in warmup_clients:
        all_client_scores.setdefault(client.client_id, [])

    lora_client_map = resume_fed_state.get("lora_client_map")
    if lora_client_map is not None:
        lora_client_map = {
            int(cluster_id): [int(client_id) for client_id in cluster_clients]
            for cluster_id, cluster_clients in lora_client_map.items()
        }
    saved_params = resume_fed_state.get("saved_params")
    optimal_n_clusters = resume_fed_state.get("optimal_n_clusters")
    aggregated_params = resume_fed_state.get("aggregated_params")
    routing_stats_history = resume_history.get("routing_stats") or {}
    task_info = resume_history.get("task_info") or task_info
    universal_idx = resume_fed_state.get("universal_idx")
    universal_init_params = resume_fed_state.get("universal_init_params")
    learned_affinity = (
        _normalize_learned_affinity_state(resume_fed_state.get("learned_affinity"))
        if needs_alpha_state
        else None
    )
    learned_client_soft_membership = None
    learned_server_soft_membership = None
    last_affinity_refresh_round = (
        learned_affinity.get("last_refresh_round")
        if learned_affinity is not None
        else None
    )
    soft_membership = resume_fed_state.get("soft_membership")
    if soft_membership is not None:
        soft_membership = {
            int(client_id): {
                int(cluster_id): float(weight)
                for cluster_id, weight in weights.items()
            }
            for client_id, weights in soft_membership.items()
        }
    static_soft_membership = (
        deepcopy(soft_membership)
        if needs_alpha_state and soft_membership is not None
        else None
    )
    if needs_alpha_state and learned_affinity is not None and lora_client_map is not None:
        client_alpha = learned_affinity.get("client_alpha")
        server_alpha = learned_affinity.get("server_alpha")
        if client_alpha is not None:
            learned_client_soft_membership = _expand_cluster_affinity_to_clients(
                client_alpha,
                _build_client_cluster_map(lora_client_map, len(warmup_clients)),
                len(warmup_clients),
                base_membership=static_soft_membership,
            )
        if server_alpha is not None:
            learned_server_soft_membership = _expand_cluster_affinity_to_clients(
                server_alpha,
                _build_client_cluster_map(lora_client_map, len(warmup_clients)),
                len(warmup_clients),
                base_membership=static_soft_membership,
            )
    per_cluster_init_params = resume_fed_state.get("per_cluster_init_params")
    if per_cluster_init_params is not None:
        per_cluster_init_params = {
            int(cluster_id): params
            for cluster_id, params in per_cluster_init_params.items()
        }
    structural_state = {
        "assignment_mode": assignment_mode,
        "soft_membership_mode": soft_membership_mode,
        "soft_membership_family_mode": soft_membership_family_mode,
        "soft_train_weight": soft_train_weight,
        "soft_floor": soft_floor,
        "visa_conflict_clip": visa_conflict_clip,
        "fedrod_dual_head": fedrod_dual_head,
        "fedrod_alpha_init": fedrod_alpha_init,
        "fedrod_universal_coeff": fedrod_universal_coeff,
        "fedrod_alpha_coeff": fedrod_alpha_coeff,
    }
    for key, cli_val in structural_state.items():
        ck_val = resume_fed_state.get(key)
        if ck_val is not None and cli_val is not None and cli_val != ck_val:
            print(
                f"[resume] WARN structural mismatch on {key}: "
                f"CLI={cli_val} vs checkpoint={ck_val}; preferring checkpoint"
            )
    if resume_fed_state.get("assignment_mode") is not None:
        assignment_mode = resume_fed_state["assignment_mode"]
    if resume_fed_state.get("soft_membership_mode") is not None:
        soft_membership_mode = resume_fed_state["soft_membership_mode"]
    if resume_fed_state.get("soft_membership_family_mode") is not None:
        soft_membership_family_mode = resume_fed_state["soft_membership_family_mode"]
    if resume_fed_state.get("soft_train_weight") is not None:
        soft_train_weight = float(resume_fed_state["soft_train_weight"])
    if resume_fed_state.get("soft_floor") is not None:
        soft_floor = float(resume_fed_state["soft_floor"])
    if resume_fed_state.get("visa_conflict_clip") is not None:
        visa_conflict_clip = bool(resume_fed_state["visa_conflict_clip"])
    if resume_fed_state.get("fedrod_dual_head") is not None:
        fedrod_dual_head = bool(resume_fed_state["fedrod_dual_head"])
    if resume_fed_state.get("fedrod_alpha_init") is not None:
        fedrod_alpha_init = float(resume_fed_state["fedrod_alpha_init"])
    if resume_fed_state.get("fedrod_universal_coeff") is not None:
        fedrod_universal_coeff = float(resume_fed_state["fedrod_universal_coeff"])
    if resume_fed_state.get("fedrod_alpha_coeff") is not None:
        fedrod_alpha_coeff = float(resume_fed_state["fedrod_alpha_coeff"])
    if resume_fed_state.get("rdrop_kl_coeff") is not None:
        rdrop_kl_coeff = float(resume_fed_state["rdrop_kl_coeff"])
    if resume_fed_state.get("rdrop_direction") is not None:
        rdrop_direction = resume_fed_state["rdrop_direction"]
    if resume_fed_state.get("rdrop_stopgrad") is not None:
        rdrop_stopgrad = resume_fed_state["rdrop_stopgrad"]
    final_adaptive = False
    final_lora_n = 1
    effective_load_balance_coeff = 0.0 if additive_residual else load_balance_coeff
    phase = resume_progress.get("phase", "warmup") if resume_state else "warmup"
    start_round_idx = int(resume_progress.get("next_round_idx", 0)) if resume_state else 0
    global_start_round_idx = (
        start_round_idx
        if phase == "warmup"
        else round_warmup + start_round_idx
    )

    enabled_aux = sum(
        bool(enabled)
        for enabled in (
            float(uemd_coeff) > 0,
            float(uemd_logit_coeff) > 0,
            float(visa_coeff) > 0,
            fedrod_dual_head,
        )
    )
    if enabled_aux > 1:
        raise ValueError("--uemd_coeff, --uemd_logit_coeff, --visa_coeff, and --fedrod_dual_head are mutually exclusive")
    if visa_coeff > 0:
        if not (universal_expert and additive_residual):
            raise ValueError("--visa_coeff requires --universal_expert --additive_residual")
        if soft_membership_mode != "task_family":
            raise ValueError("--visa_coeff requires --soft_membership task_family")
    if rdrop_kl_coeff > 0 and visa_coeff <= 0:
        raise ValueError(
            "--rdrop_kl_coeff > 0 requires --visa_coeff > 0 (the WOS dual forward is the substrate)"
        )
    if visa_conflict_clip and soft_membership_mode != "task_family":
        raise ValueError("--visa_conflict_clip requires --soft_membership task_family")
    if fedrod_dual_head:
        if not (universal_expert and additive_residual):
            raise ValueError("--fedrod_dual_head requires --universal_expert --additive_residual")
        if soft_membership_mode != "none":
            raise ValueError("--fedrod_dual_head is currently mutually exclusive with --soft_membership")
    if client_alpha_mode == "shuffled" or server_alpha_mode == "shuffled":
        if affinity_shuffle_seed is None:
            raise ValueError("--client_exposure_mode/--server_exposure_mode shuffled requires --affinity_shuffle_seed")
    if needs_signatures:
        if affinity_tau <= 0:
            raise ValueError(f"--affinity_tau must be positive, got {affinity_tau}")
        if affinity_refresh_rounds <= 0:
            raise ValueError(
                f"--affinity_refresh_rounds must be positive, got {affinity_refresh_rounds}"
            )
    if needs_alpha_state:
        if visa_coeff <= 0:
            raise ValueError(
                "--client_exposure_mode/--server_exposure_mode require --visa_coeff > 0 (the WOS substrate)"
            )
        if not (universal_expert and additive_residual):
            raise ValueError(
                "--client_exposure_mode/--server_exposure_mode require --universal_expert --additive_residual; "
                "without these, the home_cluster_only forward mode is a no-op and "
                "collected signatures would reflect routed mixtures rather than "
                "home-cluster representations"
            )
        if soft_membership_mode != "task_family":
            raise ValueError(
                "--client_exposure_mode/--server_exposure_mode require --soft_membership task_family"
            )

    if additive_residual and load_balance_coeff != 0:
        msg = (
            f"Additive residual routing is enabled; forcing load_balance_coeff from "
            f"{load_balance_coeff} to 0.0"
        )
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    if lora_client_map is not None:
        clustered_clients, final_lora_n = _initialize_clustered_clients(
            warmup_clients=warmup_clients,
            task_info=task_info,
            client_datasets=client_datasets,
            lora_client_map=lora_client_map,
            optimal_n_clusters=optimal_n_clusters,
            universal_expert=universal_expert,
            universal_idx=universal_idx,
            additive_residual=additive_residual,
            rank=rank,
            output_dir=output_dir,
            shared_lora_a=shared_lora_a,
            visa_coeff=visa_coeff,
            rdrop_kl_coeff=rdrop_kl_coeff,
            rdrop_direction=rdrop_direction,
            rdrop_stopgrad=rdrop_stopgrad,
            fedrod_dual_head=fedrod_dual_head,
            fedrod_alpha_init=fedrod_alpha_init,
            fedrod_universal_coeff=fedrod_universal_coeff,
            fedrod_alpha_coeff=fedrod_alpha_coeff,
        )
        if clustered_clients is not None:
            clients = clustered_clients
            server = Server(clients_num=len(clients))

    for round_idx in tqdm(range(global_start_round_idx, global_rounds), desc="Global Rounds"):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"\n[{current_time}] Starting Global Round {round_idx + 1}/{global_rounds}\n")

        print(f"\nGlobal Round {round_idx + 1}/{global_rounds}")

        current_phase = "warmup" if round_idx < round_warmup else "clustered"
        phase_round_idx = _phase_local_round_idx(round_idx, round_warmup)
        resuming_this_round = resume_state is not None and round_idx == global_start_round_idx
        resumed_completed_client_ids = {
            int(client_id) for client_id in (resume_progress.get("completed_client_ids") or [])
        } if resuming_this_round else set()
        resumed_partial_client_params = {
            int(client_id): params
            for client_id, params in (resume_progress.get("partial_client_params") or {}).items()
        } if resuming_this_round else {}
        resumed_tag = resume_tag if resuming_this_round else None
        skip_to_evaluation = resumed_tag == "round_post_agg"
        round_number = round_idx + 1
        collect_affinity_signatures = (
            (client_uses_dynamic_alpha or server_uses_dynamic_alpha)
            and _should_refresh_learned_affinity(
                round_number,
                round_warmup,
                affinity_refresh_rounds,
                last_affinity_refresh_round,
            )
        )
        current_round_client_signatures = {} if collect_affinity_signatures else None

        if current_phase == "warmup":
            print(f"Running warmup phase (round {round_idx + 1}/{round_warmup})")
            with open(log_file, "a") as f:
                f.write("Starting dummy client warmup\n")

            completed_client_ids = []
            partial_client_params = {}
            client_params = []

            if not skip_to_evaluation:
                dummy.load_model()
                dummy.local_training(lr=lr, epochs=local_epochs, batch_size=batch_size)
                dummy.unload_model()

                for client in tqdm(warmup_clients, desc="Client Training (Warmup)"):
                    client_id = client.client_id
                    if (
                        client_id in resumed_completed_client_ids
                        and client_id in resumed_partial_client_params
                    ):
                        params = resumed_partial_client_params[client_id]
                        partial_client_params[client_id] = params
                        completed_client_ids.append(client_id)
                        client_params.append(params)
                        if (phase_round_idx + 1) == round_warmup:
                            saved_params = saved_params or {}
                            saved_params[client_id] = params
                        if collect_affinity_signatures:
                            current_round_client_signatures[client_id] = _collect_signature_from_params(
                                client,
                                params,
                                batch_size=batch_size,
                            )
                        continue

                    with open(log_file, "a") as f:
                        f.write(f"Training Warmup Client {client_id} ({client.task_name})...\n")

                    if checkpoint_controller is not None:
                        checkpoint_controller.update_snapshot(
                            tag=f"round_in_progress_client_{client_id}",
                            progress=_build_progress(
                                next_round_idx=phase_round_idx,
                                phase="warmup",
                                completed_client_ids=completed_client_ids,
                                partial_client_params=partial_client_params,
                                active_client_id=client_id,
                            ),
                            fed_state=_build_fed_state(
                                aggregated_params=aggregated_params,
                                saved_params=saved_params,
                                lora_client_map=lora_client_map,
                                optimal_n_clusters=optimal_n_clusters,
                                universal_idx=universal_idx,
                                universal_init_params=universal_init_params,
                                soft_membership=soft_membership,
                                per_cluster_init_params=per_cluster_init_params,
                                assignment_mode=assignment_mode,
                                soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                                soft_train_weight=soft_train_weight,
                                soft_floor=soft_floor,
                                uemd_logit_coeff=uemd_logit_coeff,
                                visa_coeff=visa_coeff,
                                visa_conflict_clip=visa_conflict_clip,
                                fedrod_dual_head=fedrod_dual_head,
                                fedrod_alpha_init=fedrod_alpha_init,
                                fedrod_universal_coeff=fedrod_universal_coeff,
                                fedrod_alpha_coeff=fedrod_alpha_coeff,
                                rdrop_kl_coeff=rdrop_kl_coeff,
                                rdrop_direction=rdrop_direction,
                                rdrop_stopgrad=rdrop_stopgrad,
                                affinity_mode=affinity_mode,
                                client_alpha_mode=client_alpha_mode,
                                server_alpha_mode=server_alpha_mode,
                                learned_affinity=learned_affinity,
                            ),
                            history=_build_history_state(
                                client_scores=all_client_scores,
                                routing_stats=routing_stats_history,
                                task_info=task_info,
                            ),
                        )

                    client.load_model()
                    if round_idx > 0 and aggregated_params is not None:
                        client.load_params(aggregated_params[client_id])

                    client.local_training(lr=lr, epochs=local_epochs, batch_size=batch_size)
                    if collect_affinity_signatures:
                        current_round_client_signatures[client_id] = _collect_client_home_signature(
                            client,
                            batch_size=batch_size,
                            num_batches=8,
                        )
                    params = client.get_lora_params_and_save_by_module(
                        round_id=round_idx,
                        personal_dir=personal_dir,
                    )["params"]
                    client.unload_model()

                    partial_client_params[client_id] = params
                    completed_client_ids.append(client_id)
                    client_params.append(params)

                    if (phase_round_idx + 1) == round_warmup:
                        saved_params = saved_params or {}
                        saved_params[client_id] = params

                    if checkpoint_controller is not None:
                        # First pass: we intentionally do not serialize optimizer state.
                        # If a signal lands mid-client, we resume from the previous
                        # aggregated snapshot and retrain only the active client.
                        checkpoint_controller.maybe_save(
                            tag=f"round_in_progress_client_{client_id}",
                            progress=_build_progress(
                                next_round_idx=phase_round_idx,
                                phase="warmup",
                                completed_client_ids=completed_client_ids,
                                partial_client_params=partial_client_params,
                                active_client_id=None,
                            ),
                            fed_state=_build_fed_state(
                                aggregated_params=aggregated_params,
                                saved_params=saved_params,
                                lora_client_map=lora_client_map,
                                optimal_n_clusters=optimal_n_clusters,
                                universal_idx=universal_idx,
                                universal_init_params=universal_init_params,
                                soft_membership=soft_membership,
                                per_cluster_init_params=per_cluster_init_params,
                                assignment_mode=assignment_mode,
                                soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                                soft_train_weight=soft_train_weight,
                                soft_floor=soft_floor,
                                uemd_logit_coeff=uemd_logit_coeff,
                                visa_coeff=visa_coeff,
                                visa_conflict_clip=visa_conflict_clip,
                                fedrod_dual_head=fedrod_dual_head,
                                fedrod_alpha_init=fedrod_alpha_init,
                                fedrod_universal_coeff=fedrod_universal_coeff,
                                fedrod_alpha_coeff=fedrod_alpha_coeff,
                                rdrop_kl_coeff=rdrop_kl_coeff,
                                rdrop_direction=rdrop_direction,
                                rdrop_stopgrad=rdrop_stopgrad,
                                affinity_mode=affinity_mode,
                                client_alpha_mode=client_alpha_mode,
                                server_alpha_mode=server_alpha_mode,
                                learned_affinity=learned_affinity,
                            ),
                            history=_build_history_state(
                                client_scores=all_client_scores,
                                routing_stats=routing_stats_history,
                                task_info=task_info,
                            ),
                        )
            else:
                if len(resumed_partial_client_params) != len(warmup_clients):
                    raise RuntimeError(
                        "Warmup resume checkpoint marked as round_post_agg is missing client params"
                    )
                completed_client_ids = [client.client_id for client in warmup_clients]
                partial_client_params = dict(resumed_partial_client_params)
                client_params = [
                    partial_client_params[client.client_id] for client in warmup_clients
                ]
                if collect_affinity_signatures:
                    for client in warmup_clients:
                        current_round_client_signatures[client.client_id] = (
                            _collect_signature_from_params(
                                client,
                                partial_client_params[client.client_id],
                                batch_size=batch_size,
                            )
                        )

            with open(log_file, "a") as f:
                f.write("Starting Server Aggregation (Warmup)...\n")

            agg_lora_client_map = None
            if (phase_round_idx + 1) == round_warmup and not skip_to_evaluation:
                with open(log_file, "a") as f:
                    f.write(f"Computing LoRA client mapping (mode={assignment_mode})\n")

                forced = _build_assignment(
                    assignment_mode,
                    [task_info[i]["task_name"] for i in range(len(warmup_clients))],
                    len(warmup_clients),
                    seed=42,
                )
                if forced is not None:
                    lora_client_map, optimal_n_clusters = forced
                    with open(log_file, "a") as f:
                        f.write(f"Forced assignment ({assignment_mode}): {lora_client_map}\n")
                        f.write(f"Number of clusters: {optimal_n_clusters}\n")
                else:
                    lora_client_map, optimal_n_clusters = compute_lora_client_map(
                        warmup_clients,
                        round_idx,
                        personal_dir,
                        max_clusters=max_clusters,
                    )
                    with open(log_file, "a") as f:
                        f.write(f"Learned clustering: {lora_client_map}\n")
                        f.write(f"Optimal number of clusters: {optimal_n_clusters}\n")

                agg_lora_client_map = lora_client_map
                if universal_expert:
                    universal_idx = optimal_n_clusters
                    universal_init_params = _compute_universal_init_params(saved_params)

                if task_info is not None and client_datasets is not None:
                    clients, final_lora_n = _initialize_clustered_clients(
                        warmup_clients=warmup_clients,
                        task_info=task_info,
                        client_datasets=client_datasets,
                        lora_client_map=lora_client_map,
                        optimal_n_clusters=optimal_n_clusters,
                        universal_expert=universal_expert,
                        universal_idx=universal_idx,
                        additive_residual=additive_residual,
                        rank=rank,
                        output_dir=output_dir,
                        shared_lora_a=shared_lora_a,
                        visa_coeff=visa_coeff,
                        rdrop_kl_coeff=rdrop_kl_coeff,
                        rdrop_direction=rdrop_direction,
                        rdrop_stopgrad=rdrop_stopgrad,
                        fedrod_dual_head=fedrod_dual_head,
                        fedrod_alpha_init=fedrod_alpha_init,
                        fedrod_universal_coeff=fedrod_universal_coeff,
                        fedrod_alpha_coeff=fedrod_alpha_coeff,
                    )
                    server = Server(clients_num=len(clients))
                    with open(log_file, "a") as f:
                        f.write(
                            f"Initialized {len(clients)} clustered clients with {final_lora_n} "
                            f"LoRA modules (clusters={optimal_n_clusters}, universal_idx={universal_idx})\n"
                        )

            if not skip_to_evaluation:
                aggregated_params = server.aggregation_warmup(
                    route_aggregation=True,
                    params=client_params,
                    lora_client_map=agg_lora_client_map,
                )
                if collect_affinity_signatures:
                    (
                        learned_client_soft_membership,
                        learned_server_soft_membership,
                        learned_affinity,
                    ) = _refresh_learned_affinity_state(
                        current_round_client_signatures,
                        lora_client_map,
                        len(warmup_clients),
                        round_number,
                        affinity_tau,
                        client_alpha_mode,
                        server_alpha_mode,
                        affinity_shuffle_seed,
                        static_soft_membership,
                        log_file,
                        previous_state=learned_affinity,
                        log_shuffle_derangement=(
                            (client_alpha_mode == "shuffled" or server_alpha_mode == "shuffled")
                            and round_number == round_warmup
                        ),
                    )
                    last_affinity_refresh_round = round_number
                    _refresh_client_visa_weights(
                        clients or warmup_clients,
                        learned_client_soft_membership,
                    )
                if checkpoint_controller is not None:
                    checkpoint_controller.maybe_save(
                        tag="round_post_agg",
                        progress=_build_progress(
                            next_round_idx=phase_round_idx,
                            phase="warmup",
                            completed_client_ids=completed_client_ids,
                            partial_client_params=partial_client_params,
                            active_client_id=None,
                        ),
                        fed_state=_build_fed_state(
                            aggregated_params=aggregated_params,
                            saved_params=saved_params,
                            lora_client_map=lora_client_map,
                            optimal_n_clusters=optimal_n_clusters,
                            universal_idx=universal_idx,
                            universal_init_params=universal_init_params,
                            soft_membership=soft_membership,
                            per_cluster_init_params=per_cluster_init_params,
                            assignment_mode=assignment_mode,
                            soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                            soft_train_weight=soft_train_weight,
                            soft_floor=soft_floor,
                            uemd_logit_coeff=uemd_logit_coeff,
                            visa_coeff=visa_coeff,
                            visa_conflict_clip=visa_conflict_clip,
                            fedrod_dual_head=fedrod_dual_head,
                            fedrod_alpha_init=fedrod_alpha_init,
                            fedrod_universal_coeff=fedrod_universal_coeff,
                            fedrod_alpha_coeff=fedrod_alpha_coeff,
                            rdrop_kl_coeff=rdrop_kl_coeff,
                            rdrop_direction=rdrop_direction,
                            rdrop_stopgrad=rdrop_stopgrad,
                            affinity_mode=affinity_mode,
                            client_alpha_mode=client_alpha_mode,
                            server_alpha_mode=server_alpha_mode,
                            learned_affinity=learned_affinity,
                        ),
                        history=_build_history_state(
                            client_scores=all_client_scores,
                            routing_stats=routing_stats_history,
                            task_info=task_info,
                        ),
                    )
            elif checkpoint_controller is not None:
                if collect_affinity_signatures:
                    (
                        learned_client_soft_membership,
                        learned_server_soft_membership,
                        learned_affinity,
                    ) = _refresh_learned_affinity_state(
                        current_round_client_signatures,
                        lora_client_map,
                        len(warmup_clients),
                        round_number,
                        affinity_tau,
                        client_alpha_mode,
                        server_alpha_mode,
                        affinity_shuffle_seed,
                        static_soft_membership,
                        log_file,
                        previous_state=learned_affinity,
                        log_shuffle_derangement=(
                            (client_alpha_mode == "shuffled" or server_alpha_mode == "shuffled")
                            and round_number == round_warmup
                        ),
                    )
                    last_affinity_refresh_round = round_number
                    _refresh_client_visa_weights(
                        clients or warmup_clients,
                        learned_client_soft_membership,
                    )
                checkpoint_controller.update_snapshot(
                    tag="round_post_agg",
                    progress=_build_progress(
                        next_round_idx=phase_round_idx,
                        phase="warmup",
                        completed_client_ids=completed_client_ids,
                        partial_client_params=partial_client_params,
                        active_client_id=None,
                    ),
                    fed_state=_build_fed_state(
                        aggregated_params=aggregated_params,
                        saved_params=saved_params,
                        lora_client_map=lora_client_map,
                        optimal_n_clusters=optimal_n_clusters,
                        universal_idx=universal_idx,
                        universal_init_params=universal_init_params,
                        soft_membership=soft_membership,
                        per_cluster_init_params=per_cluster_init_params,
                        assignment_mode=assignment_mode,
                        soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                        soft_train_weight=soft_train_weight,
                        soft_floor=soft_floor,
                        uemd_logit_coeff=uemd_logit_coeff,
                        visa_coeff=visa_coeff,
                        visa_conflict_clip=visa_conflict_clip,
                        fedrod_dual_head=fedrod_dual_head,
                        fedrod_alpha_init=fedrod_alpha_init,
                        fedrod_universal_coeff=fedrod_universal_coeff,
                        fedrod_alpha_coeff=fedrod_alpha_coeff,
                        rdrop_kl_coeff=rdrop_kl_coeff,
                        rdrop_direction=rdrop_direction,
                        rdrop_stopgrad=rdrop_stopgrad,
                        affinity_mode=affinity_mode,
                        client_alpha_mode=client_alpha_mode,
                        server_alpha_mode=server_alpha_mode,
                        learned_affinity=learned_affinity,
                    ),
                    history=_build_history_state(
                        client_scores=all_client_scores,
                        routing_stats=routing_stats_history,
                        task_info=task_info,
                    ),
                )

            if (round_idx + 1) % 1 == 0:
                with open(log_file, "a") as f:
                    f.write(f"Performing warmup evaluation at round {round_idx + 1}\n")

                print(f"\nWarmup Round {round_idx + 1} Evaluation Scores:")
                round_scores = {}
                for client in warmup_clients:
                    client_id = client.client_id
                    client.load_model()
                    client.load_params(aggregated_params[client_id])
                    metrics = client.evaluate_model()
                    all_client_scores[client_id].append(metrics)
                    round_scores[client_id] = metrics
                    client.unload_model()

                summary_file = os.path.join(personal_dir, f"round_summary_{round_idx + 1}.json")
                with open(summary_file, "w") as f:
                    json.dump(round_scores, f, indent=2)

                if checkpoint_controller is not None:
                    if (phase_round_idx + 1) == round_warmup:
                        checkpoint_controller.maybe_save(
                            tag="warmup_end",
                            progress=_build_progress(
                                next_round_idx=0,
                                phase="clustered",
                                completed_client_ids=[],
                                partial_client_params={},
                                active_client_id=None,
                            ),
                            fed_state=_build_fed_state(
                                aggregated_params=aggregated_params,
                                saved_params=saved_params,
                                lora_client_map=lora_client_map,
                                optimal_n_clusters=optimal_n_clusters,
                                universal_idx=universal_idx,
                                universal_init_params=universal_init_params,
                                soft_membership=soft_membership,
                                per_cluster_init_params=per_cluster_init_params,
                                assignment_mode=assignment_mode,
                                soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                                soft_train_weight=soft_train_weight,
                                soft_floor=soft_floor,
                                uemd_logit_coeff=uemd_logit_coeff,
                                visa_coeff=visa_coeff,
                                visa_conflict_clip=visa_conflict_clip,
                                fedrod_dual_head=fedrod_dual_head,
                                fedrod_alpha_init=fedrod_alpha_init,
                                fedrod_universal_coeff=fedrod_universal_coeff,
                                fedrod_alpha_coeff=fedrod_alpha_coeff,
                                rdrop_kl_coeff=rdrop_kl_coeff,
                                rdrop_direction=rdrop_direction,
                                rdrop_stopgrad=rdrop_stopgrad,
                                affinity_mode=affinity_mode,
                                client_alpha_mode=client_alpha_mode,
                                server_alpha_mode=server_alpha_mode,
                                learned_affinity=learned_affinity,
                            ),
                            history=_build_history_state(
                                client_scores=all_client_scores,
                                routing_stats=routing_stats_history,
                                task_info=task_info,
                            ),
                        )
                    else:
                        checkpoint_controller.maybe_save(
                            tag="round_post_eval",
                            progress=_build_progress(
                                next_round_idx=phase_round_idx + 1,
                                phase="warmup",
                                completed_client_ids=[],
                                partial_client_params={},
                                active_client_id=None,
                            ),
                            fed_state=_build_fed_state(
                                aggregated_params=aggregated_params,
                                saved_params=saved_params,
                                lora_client_map=lora_client_map,
                                optimal_n_clusters=optimal_n_clusters,
                                universal_idx=universal_idx,
                                universal_init_params=universal_init_params,
                                soft_membership=soft_membership,
                                per_cluster_init_params=per_cluster_init_params,
                                assignment_mode=assignment_mode,
                                soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                                soft_train_weight=soft_train_weight,
                                soft_floor=soft_floor,
                                uemd_logit_coeff=uemd_logit_coeff,
                                visa_coeff=visa_coeff,
                                visa_conflict_clip=visa_conflict_clip,
                                fedrod_dual_head=fedrod_dual_head,
                                fedrod_alpha_init=fedrod_alpha_init,
                                fedrod_universal_coeff=fedrod_universal_coeff,
                                fedrod_alpha_coeff=fedrod_alpha_coeff,
                                rdrop_kl_coeff=rdrop_kl_coeff,
                                rdrop_direction=rdrop_direction,
                                rdrop_stopgrad=rdrop_stopgrad,
                                affinity_mode=affinity_mode,
                                client_alpha_mode=client_alpha_mode,
                                server_alpha_mode=server_alpha_mode,
                                learned_affinity=learned_affinity,
                            ),
                            history=_build_history_state(
                                client_scores=all_client_scores,
                                routing_stats=routing_stats_history,
                                task_info=task_info,
                            ),
                        )
        else:
            if clients is None:
                raise RuntimeError("Clustered clients not initialized")

            universal_warmup_end = round_warmup + max(universal_warmup_rounds, 0)
            in_universal_warmup = (
                universal_expert
                and universal_idx is not None
                and not fedrod_dual_head
                and round_idx < universal_warmup_end
            )
            use_load_balance = (
                universal_expert
                and universal_idx is not None
                and not additive_residual
                and not in_universal_warmup
                and effective_load_balance_coeff > 0
            )
            stage_name = (
                "Stage A (universal warmup)"
                if in_universal_warmup
                else "Stage B (additive residual)"
                if universal_expert and universal_idx is not None and additive_residual
                else "Stage B (full clustered training)"
                if universal_expert and universal_idx is not None
                else "Clustered training"
            )
            print(
                f"Running clustered training phase (round {round_idx + 1 - round_warmup}/{global_rounds - round_warmup}) "
                f"[{stage_name}]"
            )

            adaptive_enabled = adaptive_delay == 0 or round_idx >= (round_warmup + adaptive_delay)
            if in_universal_warmup and not adaptive_enabled:
                force_msg = (
                    f"Universal warmup overrides adaptive_delay in round {round_idx + 1}; "
                    "enabling adaptive routing so the router can learn to use the universal expert\n"
                )
                print(force_msg.strip())
                with open(log_file, "a") as f:
                    f.write(force_msg)
                adaptive_enabled = True

            use_load_balance = use_load_balance and adaptive_enabled
            final_adaptive = adaptive_enabled
            for client in clients:
                client.set_adaptive(adaptive_enabled)

            if (
                adaptive_delay > 0
                and round_idx < (round_warmup + adaptive_delay)
                and not in_universal_warmup
            ):
                with open(log_file, "a") as f:
                    f.write(
                        f"Adaptive routing delayed until round {round_warmup + adaptive_delay + 1}; "
                        f"using hard assigned-expert routing in round {round_idx + 1}\n"
                    )
            elif adaptive_delay > 0 and round_idx == (round_warmup + adaptive_delay):
                enable_msg = f"Enabling adaptive routing at round {round_idx + 1}"
                print(enable_msg)
                with open(log_file, "a") as f:
                    f.write(enable_msg + "\n")

            completed_client_ids = []
            partial_client_params = {}
            client_params = []

            if round_idx == round_warmup and not skip_to_evaluation:
                with open(log_file, "a") as f:
                    f.write("Transitioning from warmup to clustered training\n")
                    f.write(f"LoRA client mapping: {lora_client_map}\n")

                soft_membership = (
                    _build_soft_membership(
                        soft_membership_mode,
                        lora_client_map,
                        task_info,
                        num_clients=len(clients),
                        epsilon=soft_train_weight,
                        floor=soft_floor,
                        family_mode=soft_membership_family_mode,
                    )
                    if soft_membership_mode != "none"
                    else None
                )
                if needs_alpha_state:
                    static_soft_membership = (
                        deepcopy(soft_membership)
                        if soft_membership is not None
                        else None
                    )
                    if learned_affinity is None and not needs_signatures:
                        (
                            learned_client_soft_membership,
                            learned_server_soft_membership,
                            learned_affinity,
                        ) = _refresh_learned_affinity_state(
                            None,
                            lora_client_map,
                            len(clients),
                            round_number,
                            affinity_tau,
                            client_alpha_mode,
                            server_alpha_mode,
                            affinity_shuffle_seed,
                            static_soft_membership,
                            log_file,
                            previous_state=None,
                        )
                        last_affinity_refresh_round = round_number
                    if learned_affinity is not None:
                        client_to_cluster = _build_client_cluster_map(lora_client_map, len(clients))
                        client_alpha = learned_affinity.get("client_alpha")
                        server_alpha = learned_affinity.get("server_alpha")
                        learned_client_soft_membership = _expand_cluster_affinity_to_clients(
                            client_alpha,
                            client_to_cluster,
                            len(clients),
                            base_membership=static_soft_membership,
                        )
                        learned_server_soft_membership = _expand_cluster_affinity_to_clients(
                            server_alpha,
                            client_to_cluster,
                            len(clients),
                            base_membership=static_soft_membership,
                        )
                        _refresh_client_visa_weights(clients, learned_client_soft_membership)
                per_cluster_init_params = (
                    _compute_per_cluster_init_params(saved_params, lora_client_map)
                    if soft_membership is not None or needs_alpha_state
                    else None
                )

                if soft_membership is not None:
                    soft_path = os.path.join(personal_dir, "soft_membership.json")
                    with open(soft_path, "w") as f:
                        json.dump({
                            "mode": soft_membership_mode,
                            "family_mode": soft_membership_family_mode,
                            "epsilon": soft_train_weight,
                            "floor": soft_floor,
                            "membership": {
                                str(client_id): {
                                    str(cluster_id): weight
                                    for cluster_id, weight in weights.items()
                                }
                                for client_id, weights in soft_membership.items()
                            },
                        }, f, indent=2)
                    print(
                        f"[soft] Saved soft_membership.json "
                        f"(mode={soft_membership_mode}, family_mode={soft_membership_family_mode}, "
                        f"epsilon={soft_train_weight})"
                    )

                for client in clients:
                    client.load_model()
                    if saved_params is not None and client.client_id in saved_params:
                        warmed_params = _build_clustered_warm_start(
                            client,
                            saved_params[client.client_id],
                            universal_init_params=universal_init_params,
                            per_cluster_init_params=per_cluster_init_params,
                        )
                        client.local_model.load_state_dict(warmed_params, strict=False)
                    client.unload_model()

            if client_uses_side_alpha and learned_client_soft_membership is None:
                raise RuntimeError(
                    f"--client_exposure_mode {client_alpha_mode} requires client exposure distribution before clustered training"
                )
            if server_uses_side_alpha and learned_server_soft_membership is None:
                raise RuntimeError(
                    f"--server_exposure_mode {server_alpha_mode} requires server exposure distribution before clustered training"
                )

            if not skip_to_evaluation:
                for client in tqdm(clients, desc="Client Training (Clustered)"):
                    client_id = client.client_id
                    if (
                        client_id in resumed_completed_client_ids
                        and client_id in resumed_partial_client_params
                    ):
                        params = resumed_partial_client_params[client_id]
                        partial_client_params[client_id] = params
                        completed_client_ids.append(client_id)
                        client_params.append(params)
                        if collect_affinity_signatures:
                            current_round_client_signatures[client_id] = _collect_signature_from_params(
                                client,
                                params,
                                batch_size=batch_size,
                            )
                        continue

                    with open(log_file, "a") as f:
                        f.write(
                            f"Training Clustered Client {client_id} ({client.task_name}) "
                            f"[stage={stage_name}, load_balance={use_load_balance}]...\n"
                        )

                    if checkpoint_controller is not None:
                        checkpoint_controller.update_snapshot(
                            tag=f"round_in_progress_client_{client_id}",
                            progress=_build_progress(
                                next_round_idx=phase_round_idx,
                                phase="clustered",
                                completed_client_ids=completed_client_ids,
                                partial_client_params=partial_client_params,
                                active_client_id=client_id,
                            ),
                            fed_state=_build_fed_state(
                                aggregated_params=aggregated_params,
                                saved_params=saved_params,
                                lora_client_map=lora_client_map,
                                optimal_n_clusters=optimal_n_clusters,
                                universal_idx=universal_idx,
                                universal_init_params=universal_init_params,
                                soft_membership=soft_membership,
                                per_cluster_init_params=per_cluster_init_params,
                                assignment_mode=assignment_mode,
                                soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                                soft_train_weight=soft_train_weight,
                                soft_floor=soft_floor,
                                uemd_logit_coeff=uemd_logit_coeff,
                                visa_coeff=visa_coeff,
                                visa_conflict_clip=visa_conflict_clip,
                                fedrod_dual_head=fedrod_dual_head,
                                fedrod_alpha_init=fedrod_alpha_init,
                                fedrod_universal_coeff=fedrod_universal_coeff,
                                fedrod_alpha_coeff=fedrod_alpha_coeff,
                                rdrop_kl_coeff=rdrop_kl_coeff,
                                rdrop_direction=rdrop_direction,
                                rdrop_stopgrad=rdrop_stopgrad,
                                affinity_mode=affinity_mode,
                                client_alpha_mode=client_alpha_mode,
                                server_alpha_mode=server_alpha_mode,
                                learned_affinity=learned_affinity,
                            ),
                            history=_build_history_state(
                                client_scores=all_client_scores,
                                routing_stats=routing_stats_history,
                                task_info=task_info,
                            ),
                        )

                    client.load_model()
                    if round_idx > round_warmup and aggregated_params is not None:
                        client.load_params(aggregated_params[client_id])

                    if client_uses_side_alpha:
                        soft_membership_for_client = learned_client_soft_membership.get(client.client_id)
                    else:
                        soft_membership_for_client = (
                            soft_membership.get(client.client_id) if soft_membership else None
                        )

                    client.local_training(
                        lr=lr,
                        epochs=local_epochs,
                        batch_size=batch_size,
                        lora_client_map=lora_client_map,
                        in_universal_warmup=in_universal_warmup,
                        load_balance_coeff=effective_load_balance_coeff if use_load_balance else 0.0,
                        uemd_coeff=uemd_coeff,
                        uemd_logit_coeff=uemd_logit_coeff,
                        visa_coeff=visa_coeff,
                        soft_membership_for_client=soft_membership_for_client,
                        shared_lora_a=shared_lora_a,
                        fedrod_dual_head=fedrod_dual_head,
                        fedrod_universal_coeff=fedrod_universal_coeff,
                        fedrod_alpha_coeff=fedrod_alpha_coeff,
                        affinity_mode=affinity_mode,
                    )
                    if collect_affinity_signatures:
                        current_round_client_signatures[client_id] = _collect_client_home_signature(
                            client,
                            batch_size=batch_size,
                            num_batches=8,
                        )
                    params = client.get_lora_params()["params"]
                    client.unload_model()

                    partial_client_params[client_id] = params
                    completed_client_ids.append(client_id)
                    client_params.append(params)

                    if checkpoint_controller is not None:
                        checkpoint_controller.maybe_save(
                            tag=f"round_in_progress_client_{client_id}",
                            progress=_build_progress(
                                next_round_idx=phase_round_idx,
                                phase="clustered",
                                completed_client_ids=completed_client_ids,
                                partial_client_params=partial_client_params,
                                active_client_id=None,
                            ),
                            fed_state=_build_fed_state(
                                aggregated_params=aggregated_params,
                                saved_params=saved_params,
                                lora_client_map=lora_client_map,
                                optimal_n_clusters=optimal_n_clusters,
                                universal_idx=universal_idx,
                                universal_init_params=universal_init_params,
                                soft_membership=soft_membership,
                                per_cluster_init_params=per_cluster_init_params,
                                assignment_mode=assignment_mode,
                                soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                                soft_train_weight=soft_train_weight,
                                soft_floor=soft_floor,
                                uemd_logit_coeff=uemd_logit_coeff,
                                visa_coeff=visa_coeff,
                                visa_conflict_clip=visa_conflict_clip,
                                fedrod_dual_head=fedrod_dual_head,
                                fedrod_alpha_init=fedrod_alpha_init,
                                fedrod_universal_coeff=fedrod_universal_coeff,
                                fedrod_alpha_coeff=fedrod_alpha_coeff,
                                rdrop_kl_coeff=rdrop_kl_coeff,
                                rdrop_direction=rdrop_direction,
                                rdrop_stopgrad=rdrop_stopgrad,
                                affinity_mode=affinity_mode,
                                client_alpha_mode=client_alpha_mode,
                                server_alpha_mode=server_alpha_mode,
                                learned_affinity=learned_affinity,
                            ),
                            history=_build_history_state(
                                client_scores=all_client_scores,
                                routing_stats=routing_stats_history,
                                task_info=task_info,
                            ),
                        )
            else:
                if len(resumed_partial_client_params) != len(clients):
                    raise RuntimeError(
                        "Clustered resume checkpoint marked as round_post_agg is missing client params"
                    )
                completed_client_ids = [client.client_id for client in clients]
                partial_client_params = dict(resumed_partial_client_params)
                client_params = [partial_client_params[client.client_id] for client in clients]
                if collect_affinity_signatures:
                    for client in clients:
                        current_round_client_signatures[client.client_id] = (
                            _collect_signature_from_params(
                                client,
                                partial_client_params[client.client_id],
                                batch_size=batch_size,
                            )
                        )

            with open(log_file, "a") as f:
                f.write("Starting Server Aggregation (Clustered)...\n")

            if not skip_to_evaluation:
                server_soft_membership = (
                    learned_server_soft_membership
                    if server_uses_side_alpha
                    else soft_membership
                )
                if assignment_mode == "no_agg":
                    aggregated_params = server.no_aggregation(params=client_params)
                else:
                    aggregated_params = server.aggregation(
                        route_aggregation=True,
                        params=client_params,
                        lora_client_map=lora_client_map,
                        universal_idx=universal_idx,
                        soft_membership=server_soft_membership,
                        shared_lora_a=shared_lora_a,
                        fedrod_dual_head=fedrod_dual_head,
                        visa_conflict_clip=visa_conflict_clip,
                    )

                if collect_affinity_signatures:
                    (
                        learned_client_soft_membership,
                        learned_server_soft_membership,
                        learned_affinity,
                    ) = _refresh_learned_affinity_state(
                        current_round_client_signatures,
                        lora_client_map,
                        len(clients),
                        round_number,
                        affinity_tau,
                        client_alpha_mode,
                        server_alpha_mode,
                        affinity_shuffle_seed,
                        static_soft_membership,
                        log_file,
                        previous_state=learned_affinity,
                        log_shuffle_derangement=(
                            (client_alpha_mode == "shuffled" or server_alpha_mode == "shuffled")
                            and round_number == round_warmup
                        ),
                    )
                    last_affinity_refresh_round = round_number
                    _refresh_client_visa_weights(clients, learned_client_soft_membership)

                if checkpoint_controller is not None:
                    checkpoint_controller.maybe_save(
                        tag="round_post_agg",
                        progress=_build_progress(
                            next_round_idx=phase_round_idx,
                            phase="clustered",
                            completed_client_ids=completed_client_ids,
                            partial_client_params=partial_client_params,
                            active_client_id=None,
                        ),
                        fed_state=_build_fed_state(
                            aggregated_params=aggregated_params,
                            saved_params=saved_params,
                            lora_client_map=lora_client_map,
                            optimal_n_clusters=optimal_n_clusters,
                            universal_idx=universal_idx,
                            universal_init_params=universal_init_params,
                            soft_membership=soft_membership,
                            per_cluster_init_params=per_cluster_init_params,
                            assignment_mode=assignment_mode,
                            soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                            soft_train_weight=soft_train_weight,
                            soft_floor=soft_floor,
                            uemd_logit_coeff=uemd_logit_coeff,
                            visa_coeff=visa_coeff,
                            visa_conflict_clip=visa_conflict_clip,
                            fedrod_dual_head=fedrod_dual_head,
                            fedrod_alpha_init=fedrod_alpha_init,
                            fedrod_universal_coeff=fedrod_universal_coeff,
                            fedrod_alpha_coeff=fedrod_alpha_coeff,
                            rdrop_kl_coeff=rdrop_kl_coeff,
                            rdrop_direction=rdrop_direction,
                            rdrop_stopgrad=rdrop_stopgrad,
                            affinity_mode=affinity_mode,
                            client_alpha_mode=client_alpha_mode,
                            server_alpha_mode=server_alpha_mode,
                            learned_affinity=learned_affinity,
                        ),
                        history=_build_history_state(
                            client_scores=all_client_scores,
                            routing_stats=routing_stats_history,
                            task_info=task_info,
                        ),
                    )
            elif checkpoint_controller is not None:
                if collect_affinity_signatures:
                    (
                        learned_client_soft_membership,
                        learned_server_soft_membership,
                        learned_affinity,
                    ) = _refresh_learned_affinity_state(
                        current_round_client_signatures,
                        lora_client_map,
                        len(clients),
                        round_number,
                        affinity_tau,
                        client_alpha_mode,
                        server_alpha_mode,
                        affinity_shuffle_seed,
                        static_soft_membership,
                        log_file,
                        previous_state=learned_affinity,
                        log_shuffle_derangement=(
                            (client_alpha_mode == "shuffled" or server_alpha_mode == "shuffled")
                            and round_number == round_warmup
                        ),
                    )
                    last_affinity_refresh_round = round_number
                    _refresh_client_visa_weights(clients, learned_client_soft_membership)
                checkpoint_controller.update_snapshot(
                    tag="round_post_agg",
                    progress=_build_progress(
                        next_round_idx=phase_round_idx,
                        phase="clustered",
                        completed_client_ids=completed_client_ids,
                        partial_client_params=partial_client_params,
                        active_client_id=None,
                    ),
                    fed_state=_build_fed_state(
                        aggregated_params=aggregated_params,
                        saved_params=saved_params,
                        lora_client_map=lora_client_map,
                        optimal_n_clusters=optimal_n_clusters,
                        universal_idx=universal_idx,
                        universal_init_params=universal_init_params,
                        soft_membership=soft_membership,
                        per_cluster_init_params=per_cluster_init_params,
                        assignment_mode=assignment_mode,
                        soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                        soft_train_weight=soft_train_weight,
                        soft_floor=soft_floor,
                        uemd_logit_coeff=uemd_logit_coeff,
                        visa_coeff=visa_coeff,
                        visa_conflict_clip=visa_conflict_clip,
                        fedrod_dual_head=fedrod_dual_head,
                        fedrod_alpha_init=fedrod_alpha_init,
                        fedrod_universal_coeff=fedrod_universal_coeff,
                        fedrod_alpha_coeff=fedrod_alpha_coeff,
                        rdrop_kl_coeff=rdrop_kl_coeff,
                        rdrop_direction=rdrop_direction,
                        rdrop_stopgrad=rdrop_stopgrad,
                        affinity_mode=affinity_mode,
                        client_alpha_mode=client_alpha_mode,
                        server_alpha_mode=server_alpha_mode,
                        learned_affinity=learned_affinity,
                    ),
                    history=_build_history_state(
                        client_scores=all_client_scores,
                        routing_stats=routing_stats_history,
                        task_info=task_info,
                    ),
                )

            if (round_idx + 1) % 1 == 0:
                with open(log_file, "a") as f:
                    f.write(f"Performing clustered evaluation at round {round_idx + 1}\n")

                print(f"\nClustered Round {round_idx + 1} Evaluation Scores:")
                round_scores = {}
                round_routing_stats = {}
                for client in clients:
                    client_id = client.client_id
                    client.load_model()
                    client.load_params(aggregated_params[client_id])
                    metrics = client.evaluate_model()
                    routing_stats = client.get_routing_stats()
                    all_client_scores[client_id].append(metrics)
                    round_scores[client_id] = metrics
                    if routing_stats:
                        round_routing_stats[client_id] = routing_stats
                    client.unload_model()

                summary_file = os.path.join(personal_dir, f"round_summary_{round_idx + 1}.json")
                with open(summary_file, "w") as f:
                    json.dump(round_scores, f, indent=2)

                if round_routing_stats:
                    routing_summary = _summarize_routing_stats(round_routing_stats)
                    routing_stats_history[str(round_idx + 1)] = {
                        "clients": round_routing_stats,
                        "average": routing_summary,
                    }

                    routing_file = os.path.join(personal_dir, f"round_routing_{round_idx + 1}.json")
                    with open(routing_file, "w") as f:
                        json.dump(routing_stats_history[str(round_idx + 1)], f, indent=2)

                    print(f"Routing stats summary: {routing_summary}")
                    with open(log_file, "a") as f:
                        f.write(f"Routing stats summary: {routing_summary}\n")

                if checkpoint_controller is not None:
                    checkpoint_controller.maybe_save(
                        tag="round_post_eval",
                        progress=_build_progress(
                            next_round_idx=phase_round_idx + 1,
                            phase="clustered",
                            completed_client_ids=[],
                            partial_client_params={},
                            active_client_id=None,
                        ),
                        fed_state=_build_fed_state(
                            aggregated_params=aggregated_params,
                            saved_params=saved_params,
                            lora_client_map=lora_client_map,
                            optimal_n_clusters=optimal_n_clusters,
                            universal_idx=universal_idx,
                            universal_init_params=universal_init_params,
                            soft_membership=soft_membership,
                            per_cluster_init_params=per_cluster_init_params,
                            assignment_mode=assignment_mode,
                            soft_membership_mode=soft_membership_mode,
                                soft_membership_family_mode=soft_membership_family_mode,
                            soft_train_weight=soft_train_weight,
                            soft_floor=soft_floor,
                            uemd_logit_coeff=uemd_logit_coeff,
                            visa_coeff=visa_coeff,
                            visa_conflict_clip=visa_conflict_clip,
                            fedrod_dual_head=fedrod_dual_head,
                            fedrod_alpha_init=fedrod_alpha_init,
                            fedrod_universal_coeff=fedrod_universal_coeff,
                            fedrod_alpha_coeff=fedrod_alpha_coeff,
                            rdrop_kl_coeff=rdrop_kl_coeff,
                            rdrop_direction=rdrop_direction,
                            rdrop_stopgrad=rdrop_stopgrad,
                            affinity_mode=affinity_mode,
                            client_alpha_mode=client_alpha_mode,
                            server_alpha_mode=server_alpha_mode,
                            learned_affinity=learned_affinity,
                        ),
                        history=_build_history_state(
                            client_scores=all_client_scores,
                            routing_stats=routing_stats_history,
                            task_info=task_info,
                        ),
                    )

        with open(log_file, "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] Completed Global Round {round_idx + 1}/{global_rounds}\n")
            f.write("-" * 50 + "\n")

    final_task_metrics = {}
    if task_info is not None:
        task_accs = {}
        for client_id, scores_list in all_client_scores.items():
            if scores_list:
                final_acc = scores_list[-1].get("eval_accuracy", 0)
                task_name = task_info[client_id]["task_name"]
                if task_name not in task_accs:
                    task_accs[task_name] = []
                task_accs[task_name].append(final_acc * 100)
        for task_name, accs in task_accs.items():
            avg = sum(accs) / len(accs)
            final_task_metrics[task_name] = avg
            print(f"  {task_name}: {avg:.2f}%")
        overall_avg = (
            sum(final_task_metrics.values()) / len(final_task_metrics) if final_task_metrics else 0
        )
        final_task_metrics["average"] = overall_avg
        print(f"  Average: {overall_avg:.2f}%")

    training_history = {
        "client_scores": all_client_scores,
        "optimal_n_clusters": optimal_n_clusters,
        "lora_client_map": {str(k): v for k, v in lora_client_map.items()} if lora_client_map else None,
        "final_task_metrics": final_task_metrics,
        "adaptive_delay": adaptive_delay,
        "soft_membership_mode": soft_membership_mode,
        "soft_membership_family_mode": soft_membership_family_mode,
        "soft_train_weight": soft_train_weight,
        "soft_floor": soft_floor,
        "client_alpha_mode": client_alpha_mode,
        "server_alpha_mode": server_alpha_mode,
        "routing_stats": routing_stats_history,
        "task_info": task_info,
        "universal_expert": universal_expert,
        "additive_residual": additive_residual,
        "universal_idx": universal_idx,
        "universal_warmup_rounds": universal_warmup_rounds,
        "load_balance_coeff": effective_load_balance_coeff,
        "uemd_coeff": uemd_coeff,
        "uemd_logit_coeff": uemd_logit_coeff,
        "visa_coeff": visa_coeff,
        "visa_conflict_clip": visa_conflict_clip,
        "fedrod_dual_head": fedrod_dual_head,
        "fedrod_alpha_init": fedrod_alpha_init,
        "fedrod_universal_coeff": fedrod_universal_coeff,
        "fedrod_alpha_coeff": fedrod_alpha_coeff,
        "rdrop_kl_coeff": rdrop_kl_coeff,
        "rdrop_direction": rdrop_direction,
        "rdrop_stopgrad": rdrop_stopgrad,
        "final_lora_n": final_lora_n,
        "final_adaptive": final_adaptive,
        "global_rounds": global_rounds,
        "warmup_rounds": round_warmup,
        "batch_size": batch_size,
        "rank": rank,
    }

    training_history_path = os.path.join(personal_dir, "training_history.json")
    with open(training_history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    if save_final_params and aggregated_params is not None:
        torch.save(_to_cpu_state(aggregated_params), os.path.join(output_dir, "final_params.pt"))

    if cross_eval and aggregated_params is not None:
        final_clients = clients if clients is not None else warmup_clients
        _run_cross_task_evaluation(
            final_clients=final_clients,
            aggregated_params=aggregated_params,
            task_info=task_info,
            lora_client_map=lora_client_map,
            output_dir=output_dir,
        )

    if MLFLOW_AVAILABLE:
        mlflow_run_id = os.environ.get("MLFLOW_RUN_ID")
        if mlflow_run_id:
            mlflow.start_run(run_id=mlflow_run_id)
        for task_name, acc in final_task_metrics.items():
            mlflow.log_metric(f"final/{task_name}_acc", acc)
        mlflow.log_artifact(training_history_path)

    return all_client_scores


def _parse_affinity_mode(value):
    if value not in {"off", "train_only", "full", "shuffled"}:
        raise argparse.ArgumentTypeError(
            f"invalid affinity mode: {value!r}"
        )
    return value


def _resolve_alpha_modes(
    affinity_mode,
    client_alpha=None,
    server_alpha=None,
    warn_explicit_overrides=False,
):
    if affinity_mode not in AFFINITY_MODE_DERIVATION:
        raise ValueError(f"Unknown affinity_mode: {affinity_mode}")

    implied_client, implied_server = AFFINITY_MODE_DERIVATION[affinity_mode]
    client_alpha_mode = client_alpha if client_alpha is not None else implied_client
    server_alpha_mode = server_alpha if server_alpha is not None else implied_server

    for side, explicit_value, implied_value in (
        ("client", client_alpha, implied_client),
        ("server", server_alpha, implied_server),
    ):
        if (
            warn_explicit_overrides
            and explicit_value is not None
            and explicit_value != implied_value
        ):
            print(
                f"WARNING: --{side}_exposure_mode={explicit_value} overrides the value "
                f"({implied_value}) implied by --affinity_mode={affinity_mode}.",
                file=sys.stderr,
            )

    return client_alpha_mode, server_alpha_mode


def _alpha_mode_predicates(client_alpha_mode, server_alpha_mode):
    client_uses_side_alpha = client_alpha_mode != "static"
    server_uses_side_alpha = server_alpha_mode != "static"
    client_uses_dynamic_alpha = client_alpha_mode in {"learned", "shuffled"}
    server_uses_dynamic_alpha = server_alpha_mode in {"learned", "shuffled"}
    return {
        "client_uses_side_alpha": client_uses_side_alpha,
        "server_uses_side_alpha": server_uses_side_alpha,
        "client_uses_dynamic_alpha": client_uses_dynamic_alpha,
        "server_uses_dynamic_alpha": server_uses_dynamic_alpha,
        "needs_signatures": client_uses_dynamic_alpha or server_uses_dynamic_alpha,
        "needs_alpha_state": client_uses_side_alpha or server_uses_side_alpha,
    }


def _argv_has_option(argv, option_name):
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FedLEASE: Adaptive LoRA Experts Allocation and Selection")
    
    parser.add_argument("--model_name", type=str, default="roberta-large",
                        help="Pre-trained model name")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "sst2", "sst2", "sst2", 
                                                         "qnli", "qnli", "qnli", "qnli",
                                                         "mrpc", "mrpc", "mrpc", "mrpc",
                                                         "qqp", "qqp", "qqp", "qqp"],
                        help="List of tasks for each client")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--global_rounds", type=int, default=25,
                        help="Number of global federated rounds")
    parser.add_argument("--local_epochs", type=int, default=2,
                        help="Number of local training epochs")
    parser.add_argument("--warmup_rounds", type=int, default=5,
                        help="Number of warmup rounds before clustering")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=4,
                        help="LoRA rank")
    parser.add_argument("--max_clusters", type=int, default=4,
                        help="Maximum number of LoRA expert clusters")
    parser.add_argument("--train_samples", type=int, default=1000,
                        help="Training samples per client")
    parser.add_argument("--test_samples", type=int, default=200,
                        help="Test samples per client")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--assignment_mode", type=str, default="learned",
                        choices=["learned", "oracle", "random", "single", "no_agg"],
                        help="Expert assignment mode for diagnostic experiments. "
                             "Robustness (additive-residual, rank=4, baseline offdiag=47.23): "
                             "exp_15 oracle seed=42 -> offdiag=47.59; "
                             "exp_29 random seed=42 -> offdiag=53.14 (+5.91, but C1 fail -4.52). "
                             "exp_31 random seed=43 -> replication under test. "
                             "exp_30 single seed=42 -> FedAvg floor in_dist=79.22.")
    parser.add_argument("--soft_membership", type=str, default="none",
                        choices=["none", "task_family", "distance", "shuffled", "all_uniform"],
                        help="Soft cluster membership mode (default: none = hard 1-of-K). "
                             "shuffled/all_uniform are aliases for task_family with the corresponding "
                             "--soft_membership_family_mode.")
    parser.add_argument("--soft_membership_family_mode", type=str, default="oracle",
                        choices=["oracle", "shuffled", "all_uniform"],
                        help="Family labels used by --soft_membership task_family. oracle uses "
                             "GLUE_TASK_FAMILIES; shuffled uses fixed seed=7 mapping "
                             "sst2->paraphrase, qnli/mnli/rte->sentiment, mrpc/qqp->nli; "
                             "all_uniform ignores family labels.")
    parser.add_argument("--soft_train_weight", type=float, default=0.20,
                        help="Epsilon: total mass given to non-home clusters (default: 0.20)")
    parser.add_argument("--soft_floor", type=float, default=0.20,
                        help="Fraction of soft_train_weight assigned to out-of-family clusters (default: 0.20)")
    parser.add_argument("--affinity_mode", type=_parse_affinity_mode, default="off",
                        choices=["off", "train_only", "full", "shuffled"],
                        help="Learned-affinity mode: off keeps the static prior, "
                             "train_only uses learned affinity only for client training, "
                             "full uses it for training and aggregation, and shuffled "
                             "uses a fixed derangement control.")
    parser.add_argument("--client_exposure_mode", "--client_alpha",
                        type=str, default=None, dest="client_alpha",
                        choices=ALPHA_MODE_CHOICES,
                        help="Exposure-distribution mode q_c(k) consumed by client-side "
                             "visa-pass weights. Defaults to value derived from --affinity_mode. "
                             "Legacy alias: --client_alpha (same dest).")
    parser.add_argument("--server_exposure_mode", "--server_alpha",
                        type=str, default=None, dest="server_alpha",
                        choices=ALPHA_MODE_CHOICES,
                        help="Exposure-distribution mode q_c(k) consumed by server-side cluster "
                             "expert aggregation. Defaults to value derived from --affinity_mode. "
                             "Legacy alias: --server_alpha (same dest).")
    parser.add_argument("--affinity_tau", type=float, default=0.5,
                        help="Temperature for learned affinity weights (default: 0.5)")
    parser.add_argument("--affinity_refresh_rounds", type=int, default=5,
                        help="Refresh interval in clustered rounds for learned affinity (default: 5)")
    parser.add_argument("--affinity_shuffle_seed", type=int, default=7,
                        help="Seed for shuffled learned-affinity derangement (default: 7)")
    parser.add_argument("--shared_lora_a", action="store_true",
                        help="HydraFed-LEASE: aggregate lora_A_k across ALL clients (not per-cluster) and train all "
                             "lora_A_k on every client. lora_B_k still per-cluster. Captures cross-task common "
                             "subspace in A. Composes with --soft_membership; standalone defaults if soft_membership=none.")
    parser.add_argument("--adaptive_delay", type=int, default=0,
                        help="Number of clustered rounds to keep hard assigned-expert routing before adaptive routing")
    parser.add_argument("--save_final_params", action=argparse.BooleanOptionalAction, default=True,
                        help="Save the final aggregated client parameter states to final_params.pt")
    parser.add_argument("--cross_eval", action="store_true",
                        help="Evaluate each final client on the full validation split of every task")
    parser.add_argument("--universal_expert", action="store_true",
                        help="Add a global universal expert on top of the clustered experts after warmup")
    parser.add_argument("--universal_warmup_rounds", type=int, default=2,
                        help="Number of clustered rounds to train only the universal B matrix while freezing cluster B matrices. "
                             "Sweep (seed=42, additive-residual, extended-GLUE, baseline in_dist=86.58 / offdiag=47.23): "
                             "exp_15 warmup=2 -> in_dist=86.81, offdiag=47.59 (Δ_off=+0.36); "
                             "exp_23 warmup=3 -> in_dist=84.75, offdiag=47.96 (Δ_off=+0.73, C1 fail); "
                             "exp_22 warmup=5 -> in_dist=84.75, offdiag=49.35 (Δ_off=+2.12, C1 fail). "
                             "Threshold effect between warmup=2 and warmup=3; non-monotonic Δ_off. "
                             "Seed variance to be characterized at warmup=5.")
    parser.add_argument("--load_balance_coeff", type=float, default=0.01,
                        help="Auxiliary load balancing coefficient for universal-expert Stage B training")
    parser.add_argument("--uemd_coeff", type=float, default=0.0,
                        help="Auxiliary Universal-Expert Mutual Distillation MSE coefficient. "
                             "Requires --universal_expert --additive_residual.")
    parser.add_argument("--uemd_logit_coeff", type=float, default=0.0,
                        help="Auxiliary Universal-Expert Mutual Distillation logit-KL coefficient. "
                             "Requires --universal_expert --additive_residual. Mutually exclusive with --uemd_coeff.")
    parser.add_argument("--visa_coeff", type=float, default=0.0,
                        help="Auxiliary write-only soft-membership CE coefficient. "
                             "Requires --universal_expert --additive_residual "
                             "and --soft_membership task_family. Mutually exclusive with UEMD losses.")
    parser.add_argument("--visa_conflict_clip", action="store_true",
                        help="Clip anti-home components from foreign lora_B soft-membership updates during "
                             "server aggregation. Requires --soft_membership task_family.")
    parser.add_argument("--rdrop_kl_coeff", type=float, default=0.0,
                        help="WOS-RDrop KL consistency coefficient between home_cluster_only and non_home_visa "
                             "output distributions (NeurIPS 2021 R-Drop adapted to the WOS dual forward). "
                             "Requires --visa_coeff > 0. Default 0.0 (off, bit-identical to no flag).")
    parser.add_argument("--rdrop_direction", type=str, default="symmetric",
                        choices=["symmetric", "home_to_visa", "visa_to_home"],
                        help="Direction of the WOS-RDrop KL term. "
                             "home_to_visa: F.kl_div(log_softmax(home), softmax(visa)) — gradient pushes home toward visa. "
                             "visa_to_home: the reverse. symmetric: 0.5*(home_to_visa + visa_to_home).")
    parser.add_argument("--rdrop_stopgrad", type=str, default="target",
                        choices=["target", "none", "both"],
                        help="Stop-gradient policy on the target side of each WOS-RDrop KL term. "
                             "target (default): detach the right-hand-side target distribution — recommended on this "
                             "WOS substrate to keep the visa branch from being pulled toward home and killing the "
                             "exposure-regularization signal that the side-isolated α panel revealed. "
                             "none: no detach (matches original R-Drop NeurIPS 2021). "
                             "both: detach both sides (the KL term has no gradient — pure ablation control).")
    parser.add_argument("--fedrod_dual_head", action="store_true",
                        help="Enable FedRoD-style dual-head additive-residual training: cluster-only logits use "
                             "the existing classifier, universal-only logits use cls_universal, and evaluation "
                             "mixes both paths with per-cluster alpha_logit.")
    parser.add_argument("--fedrod_alpha_init", type=float, default=2.0,
                        help="Initial alpha logit for FedRoD cluster/universal logit mixing "
                             "(default 2.0 gives sigmoid(alpha) ~= 0.88).")
    parser.add_argument("--fedrod_universal_coeff", type=float, default=1.0,
                        help="FedRoD universal-path CE coefficient.")
    parser.add_argument("--fedrod_alpha_coeff", type=float, default=0.0,
                        help="FedRoD mixed-logit CE coefficient for learning alpha_logit.")
    parser.add_argument("--additive_residual", action="store_true",
                        help="Use MoCLE-style additive residual routing for the universal expert. "
                             "Robustness sensitivity (seed=42, extended-GLUE, additive-residual, baseline in_dist=86.58 / offdiag=47.23): "
                             "exp_15 rank=4 -> in_dist=86.81, offdiag=47.59 (incumbent, Δ_off=+0.36); "
                             "exp_27 rank=8 -> in_dist=86.06, offdiag=48.12 (Δ_off=+0.89, best of run). "
                             "Sweeping rank=2 next to complete the sensitivity curve.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a FedLEASE checkpoint to resume from; defaults to $SLURMLAB_RESUME_CKPT when set")

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    client_alpha_mode, server_alpha_mode = _resolve_alpha_modes(
        args.affinity_mode,
        client_alpha=args.client_alpha,
        server_alpha=args.server_alpha,
        warn_explicit_overrides=_argv_has_option(raw_argv, "--affinity_mode"),
    )
    args.client_alpha_mode = client_alpha_mode
    args.server_alpha_mode = server_alpha_mode
    for key, value in _alpha_mode_predicates(client_alpha_mode, server_alpha_mode).items():
        setattr(args, key, value)

    if args.soft_membership in {"shuffled", "all_uniform"}:
        if args.soft_membership_family_mode != "oracle":
            raise ValueError(
                "Use either --soft_membership shuffled/all_uniform or "
                "--soft_membership task_family --soft_membership_family_mode <mode>, not both"
            )
        args.soft_membership_family_mode = args.soft_membership
        args.soft_membership = "task_family"
    if args.soft_membership != "none":
        if args.assignment_mode != "oracle":
            raise ValueError("--soft_membership requires --assignment_mode oracle")
        if not (0.0 <= args.soft_train_weight <= 0.5):
            raise ValueError(
                f"--soft_train_weight must be in [0.0, 0.5], got {args.soft_train_weight}"
            )
        if not (0.0 <= args.soft_floor <= 1.0):
            raise ValueError(f"--soft_floor must be in [0.0, 1.0], got {args.soft_floor}")
    if args.uemd_coeff < 0:
        raise ValueError(f"--uemd_coeff must be non-negative, got {args.uemd_coeff}")
    if args.uemd_logit_coeff < 0:
        raise ValueError(f"--uemd_logit_coeff must be non-negative, got {args.uemd_logit_coeff}")
    if args.visa_coeff < 0:
        raise ValueError(f"--visa_coeff must be non-negative, got {args.visa_coeff}")
    if args.rdrop_kl_coeff < 0:
        raise ValueError(f"--rdrop_kl_coeff must be non-negative, got {args.rdrop_kl_coeff}")
    if args.rdrop_kl_coeff > 0 and args.visa_coeff <= 0:
        raise ValueError(
            "--rdrop_kl_coeff > 0 requires --visa_coeff > 0 (the WOS dual forward is the substrate)"
        )
    if args.visa_conflict_clip and args.soft_membership != "task_family":
        raise ValueError("--visa_conflict_clip requires --soft_membership task_family")
    if args.fedrod_universal_coeff < 0:
        raise ValueError(f"--fedrod_universal_coeff must be non-negative, got {args.fedrod_universal_coeff}")
    if args.fedrod_alpha_coeff < 0:
        raise ValueError(f"--fedrod_alpha_coeff must be non-negative, got {args.fedrod_alpha_coeff}")
    enabled_aux = sum(
        bool(enabled)
        for enabled in (
            args.uemd_coeff > 0,
            args.uemd_logit_coeff > 0,
            args.visa_coeff > 0,
            args.fedrod_dual_head,
        )
    )
    if enabled_aux > 1:
        raise ValueError("--uemd_coeff, --uemd_logit_coeff, --visa_coeff, and --fedrod_dual_head are mutually exclusive")
    if args.fedrod_dual_head:
        if not (args.universal_expert and args.additive_residual):
            raise ValueError("--fedrod_dual_head requires --universal_expert --additive_residual")
        if args.soft_membership != "none":
            raise ValueError("--fedrod_dual_head is currently mutually exclusive with --soft_membership")

    return args


def _set_initial_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)


def _run_training(args):
    _ensure_project_imports()

    if args.additive_residual and not args.universal_expert:
        raise ValueError("--additive_residual requires --universal_expert")
    enabled_aux = sum(
        bool(enabled)
        for enabled in (
            args.uemd_coeff > 0,
            args.uemd_logit_coeff > 0,
            args.visa_coeff > 0,
            args.fedrod_dual_head,
        )
    )
    if enabled_aux > 1:
        raise ValueError("--uemd_coeff, --uemd_logit_coeff, --visa_coeff, and --fedrod_dual_head are mutually exclusive")
    if args.uemd_coeff > 0 and not (args.universal_expert and args.additive_residual):
        raise ValueError("--uemd_coeff requires --universal_expert --additive_residual")
    if args.uemd_logit_coeff > 0 and not (args.universal_expert and args.additive_residual):
        raise ValueError("--uemd_logit_coeff requires --universal_expert --additive_residual")
    if args.visa_coeff > 0:
        if not (args.universal_expert and args.additive_residual):
            raise ValueError("--visa_coeff requires --universal_expert --additive_residual")
        if args.soft_membership != "task_family":
            raise ValueError("--visa_coeff requires --soft_membership task_family")
    if args.rdrop_kl_coeff > 0 and args.visa_coeff <= 0:
        raise ValueError(
            "--rdrop_kl_coeff > 0 requires --visa_coeff > 0 (the WOS dual forward is the substrate)"
        )
    if args.visa_conflict_clip and args.soft_membership != "task_family":
        raise ValueError("--visa_conflict_clip requires --soft_membership task_family")
    if args.fedrod_dual_head:
        if not (args.universal_expert and args.additive_residual):
            raise ValueError("--fedrod_dual_head requires --universal_expert --additive_residual")
        if args.soft_membership != "none":
            raise ValueError("--fedrod_dual_head is currently mutually exclusive with --soft_membership")
    if args.client_alpha_mode == "shuffled" or args.server_alpha_mode == "shuffled":
        if args.affinity_shuffle_seed is None:
            raise ValueError("--client_exposure_mode/--server_exposure_mode shuffled requires --affinity_shuffle_seed")
    if args.needs_signatures:
        if args.affinity_tau <= 0:
            raise ValueError(f"--affinity_tau must be positive, got {args.affinity_tau}")
        if args.affinity_refresh_rounds <= 0:
            raise ValueError(
                f"--affinity_refresh_rounds must be positive, got {args.affinity_refresh_rounds}"
            )
    if args.needs_alpha_state:
        if args.visa_coeff <= 0:
            raise ValueError(
                "--client_exposure_mode/--server_exposure_mode require --visa_coeff > 0 (the WOS substrate)"
            )
        if not (args.universal_expert and args.additive_residual):
            raise ValueError(
                "--client_exposure_mode/--server_exposure_mode require --universal_expert --additive_residual; "
                "without these, the home_cluster_only forward mode is a no-op and "
                "collected signatures would reflect routed mixtures rather than "
                "home-cluster representations"
            )
        if args.soft_membership != "task_family":
            raise ValueError(
                "--client_exposure_mode/--server_exposure_mode require --soft_membership task_family"
            )

    resume_from = args.resume_from or os.environ.get("SLURMLAB_RESUME_CKPT")
    resume_state = _load_training_checkpoint(resume_from) if resume_from else None
    if resume_state is not None:
        _validate_resume_args(args, resume_state)
        _restore_rng_state(resume_state.get("rng"))
        print(f"[resume] loaded checkpoint {resume_from}")
    else:
        _set_initial_seeds(args.seed)

    task_name_list = args.tasks
    client_num = len(task_name_list)
    output_dir = _build_output_dir(args)
    run_dir = _resolve_run_dir(output_dir)
    # Artifact isolation: when running under Slurm with SLURMLAB_OUTPUT_DIR set
    # AND the user did not override --output_dir, route training artifacts
    # (training_history, cross_eval_results, soft_membership) to the per-run
    # output dir to avoid collisions between overlapping slab jobs sharing
    # the same default ./output path. Detected by SLURMLAB_OUTPUT_DIR being
    # set, which is only true under slab-managed execution.
    slurm_output_dir = os.environ.get("SLURMLAB_OUTPUT_DIR")
    if slurm_output_dir and args.output_dir == "./output":
        output_dir = run_dir
    checkpoint_controller = CheckpointController(run_dir=run_dir, args=args)
    checkpoint_controller.install()

    print(f"Running federated learning with multi-task datasets: {task_name_list}")
    print(f"Number of clients: {client_num}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Run directory: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    client_datasets, task_info = partition_multi_task_dataset(
        task_name_list=task_name_list,
        tokenizer=tokenizer,
        alpha=100000,
        train_samples_per_client=args.train_samples,
        test_samples_per_client=args.test_samples,
        seed=args.seed,
    )

    if resume_state is not None:
        # partition_multi_task_dataset() reseeds numpy/python internally. Restore the
        # checkpoint RNG again so training continues from the saved stochastic state.
        _restore_rng_state(resume_state.get("rng"))

    dummy_task = task_name_list[0]
    dummy_num_labels = task_info[0]["num_labels"]

    dummy = WarmupClient(
        client_id=client_num,
        task_name=dummy_task,
        tokenizer=tokenizer,
        model_name=args.model_name,
        num_clients=client_num,
        rank=args.rank,
        cache_path=output_dir,
    )
    dummy.set_dataset(client_datasets[0], dummy_num_labels)

    warmup_clients = []
    for client_id in range(client_num):
        client_task = task_info[client_id]["task_name"]
        num_labels = task_info[client_id]["num_labels"]

        client = WarmupClient(
            client_id=client_id,
            task_name=client_task,
            tokenizer=tokenizer,
            model_name=args.model_name,
            num_clients=client_num,
            rank=args.rank,
            cache_path=output_dir,
        )
        client.set_dataset(client_datasets[client_id], num_labels)
        warmup_clients.append(client)

    warmup_server = Server(clients_num=len(warmup_clients))

    train_result = train_federated(
        dummy=dummy,
        clients=warmup_clients,
        server=warmup_server,
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        output_dir=output_dir,
        lr=args.lr,
        round_warmup=args.warmup_rounds,
        max_clusters=args.max_clusters,
        assignment_mode=args.assignment_mode,
        adaptive_delay=args.adaptive_delay,
        task_info=task_info,
        client_datasets=client_datasets,
        batch_size=args.batch_size,
        rank=args.rank,
        save_final_params=args.save_final_params,
        cross_eval=args.cross_eval,
        universal_expert=args.universal_expert,
        universal_warmup_rounds=args.universal_warmup_rounds,
        load_balance_coeff=args.load_balance_coeff,
        uemd_coeff=args.uemd_coeff,
        uemd_logit_coeff=args.uemd_logit_coeff,
        visa_coeff=args.visa_coeff,
        visa_conflict_clip=args.visa_conflict_clip,
        fedrod_dual_head=args.fedrod_dual_head,
        fedrod_alpha_init=args.fedrod_alpha_init,
        fedrod_universal_coeff=args.fedrod_universal_coeff,
        fedrod_alpha_coeff=args.fedrod_alpha_coeff,
        rdrop_kl_coeff=args.rdrop_kl_coeff,
        rdrop_direction=args.rdrop_direction,
        rdrop_stopgrad=args.rdrop_stopgrad,
        additive_residual=args.additive_residual,
        soft_membership_mode=args.soft_membership,
        soft_membership_family_mode=args.soft_membership_family_mode,
        soft_train_weight=args.soft_train_weight,
        soft_floor=args.soft_floor,
        affinity_mode=args.affinity_mode,
        client_alpha_mode=args.client_alpha_mode,
        server_alpha_mode=args.server_alpha_mode,
        affinity_tau=args.affinity_tau,
        affinity_refresh_rounds=args.affinity_refresh_rounds,
        affinity_shuffle_seed=args.affinity_shuffle_seed,
        shared_lora_a=args.shared_lora_a,
        resume_state=resume_state,
        checkpoint_controller=checkpoint_controller,
    )

    print("\nTraining completed!")
    print("Final Evaluation Scores for each client:", train_result)

    _write_autoresearch_metrics(output_dir)
    return train_result


class FedLEASETrainingTask:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        return _run_training(self.args)

    def checkpoint(self, *task_args, **task_kwargs):
        if submitit is None:
            raise RuntimeError("submitit is required for FedLEASETrainingTask.checkpoint()")

        next_args = argparse.Namespace(**vars(self.args))
        if not next_args.resume_from:
            next_args.resume_from = os.environ.get("SLURMLAB_RESUME_CKPT")
        if not next_args.resume_from:
            run_dir = os.environ.get("SLURMLAB_OUTPUT_DIR")
            if run_dir:
                pointer_path = os.path.join(run_dir, CHECKPOINT_DIRNAME, CHECKPOINT_POINTER_NAME)
                if os.path.exists(pointer_path):
                    next_args.resume_from = pointer_path
        return submitit.helpers.DelayedSubmission(FedLEASETrainingTask(next_args))


def main():
    args = parse_args()
    if os.environ.get("SLURMLAB_OUTPUT_DIR") and submitit is not None:
        return FedLEASETrainingTask(args)()
    return FedLEASETrainingTask(args)()


if __name__ == "__main__":
    main()
