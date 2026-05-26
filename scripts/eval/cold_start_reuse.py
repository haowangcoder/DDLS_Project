#!/usr/bin/env python
"""New-client cold-start expert-reuse experiment.

Tests whether WOS-trained federated LoRA experts are better *reuse substrates*
than baseline-trained experts: a new client arrives with a task unseen during
training and scarce labels, freezes a reused cluster expert as a feature
extractor, and trains only a cheap linear classification head.

This file is built in two change units:
  - CU1a (this commit): GPU-free library portion -- novel-task data loading,
    stratified subsampling, the linear probe, metrics, and paired-delta stats.
  - CU1b (next commit): checkpoint loading, forced single-expert routing,
    frozen feature extraction, keyed feature cache, and the main() pipeline.

It does NOT modify any locked file (head_swap_eval.py, peft/, Phase 12).
"""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

# sys.path bootstrap must run before importing head_swap_eval / client.
ROOT = Path(__file__).resolve().parents[2]  # repo root (scripts/eval/<file>)
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from client import _get_sequence_classifier_model  # noqa: E402
from head_swap_eval import (  # noqa: E402
    _build_client,
    _build_client_cluster_map,
    _build_cluster_metadata,
    _client_params,
    _forced_cluster_routing,
    _free_client_model,
    _load_final_params,
    _load_latest_checkpoint,
    _load_run_config,
    _normalize_lora_client_map,
    _normalize_task_info,
    _resolve_training_history_path,
)


# ----------------------------------------------------------------------------
# Pre-registered protocol constants (fixed; identical across every condition).
# ----------------------------------------------------------------------------
PROBE_C = 1.0  # sklearn LogisticRegression inverse-L2 strength
PROBE_MAX_ITER = 2000  # solver iterations; solve to convergence, no early stop
BUDGETS = (16, 32, 64, 128, 500, "full")  # novel-task train-example budgets
FEW_SHOT_BUDGETS = (16, 32, 64, 128, 500)  # budgets that get subsampling
N_SUBSAMPLES = 10  # stratified subsamples per few-shot budget
INNER_SPLIT_FRAC = 0.30  # stratified inner split fraction for expert selection
COLA_MAX_LENGTH = 128
BOOLQ_MAX_LENGTH = 512

# Novel tasks: held out of federated training (training tasks were
# sst2/qnli/mrpc/qqp/mnli/rte). CoLA primary metric is MCC (class-imbalanced);
# BoolQ primary metric is accuracy (official SuperGLUE metric, ~62/38 split).
NOVEL_TASKS = ("cola", "boolq")
PRIMARY_METRIC = {"cola": "mcc", "boolq": "accuracy"}


# ----------------------------------------------------------------------------
# Novel-task dataset loading
# ----------------------------------------------------------------------------
@dataclass
class NovelTaskData:
    """Tokenized train + validation splits for one novel task."""

    task: str
    train: object  # HF Dataset with input_ids / attention_mask / label
    validation: object
    num_labels: int
    max_length: int
    truncation_rate: float  # fraction of examples whose tokens were truncated


def _measure_truncation_rate(dataset, tokenizer, text_a_key, text_b_key, max_length):
    """Fraction of examples whose *untruncated* token length exceeds max_length.

    Computed in a direct batched pass (NOT via ``Dataset.map``) so it is never
    skipped by HF's map cache -- the rate is experiment metadata and must be
    correct on every run, including cached re-runs.
    """
    total = len(dataset)
    if total == 0:
        return 0.0
    truncated = 0
    for start in range(0, total, 1000):
        chunk = dataset[start:start + 1000]
        if text_b_key is None:
            enc = tokenizer(chunk[text_a_key])
        else:
            enc = tokenizer(chunk[text_a_key], chunk[text_b_key])
        truncated += sum(len(ids) > max_length for ids in enc["input_ids"])
    return truncated / total


def _tokenize_split(dataset, tokenizer, text_a_key, text_b_key, max_length):
    """Tokenize one split; return (tokenized_dataset, truncation_rate).

    Truncation happens on the second segment only (``truncation="only_second"``)
    when a second segment exists, so the first segment (e.g. the BoolQ question)
    is always kept intact. The truncation rate is measured separately via
    ``_measure_truncation_rate`` -- not as a side effect of the cached map.
    """
    truncation = "only_second" if text_b_key is not None else True

    def _tok(examples):
        if text_b_key is None:
            return tokenizer(
                examples[text_a_key], truncation=True, max_length=max_length
            )
        return tokenizer(
            examples[text_a_key],
            examples[text_b_key],
            truncation=truncation,
            max_length=max_length,
        )

    # remove raw text / idx columns so the tokenized split contains only
    # {input_ids, attention_mask, label} -- safe to batch with a data collator.
    tokenized = dataset.map(
        _tok,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "label"],
        desc=f"tokenize({text_a_key})",
    )
    truncation_rate = _measure_truncation_rate(
        dataset, tokenizer, text_a_key, text_b_key, max_length
    )
    return tokenized, truncation_rate


def load_novel_task(task: str, tokenizer) -> NovelTaskData:
    """Load + tokenize a held-out novel task (CoLA or BoolQ)."""
    task = task.lower()
    if task == "cola":
        raw = load_dataset("glue", "cola")
        text_a, text_b, max_length = "sentence", None, COLA_MAX_LENGTH
        val_key = "validation"
    elif task == "boolq":
        raw = load_dataset("super_glue", "boolq")
        # question first (kept intact), passage second (truncated if long)
        text_a, text_b, max_length = "question", "passage", BOOLQ_MAX_LENGTH
        val_key = "validation"
    else:
        raise ValueError(f"Unknown novel task: {task!r} (expected cola/boolq)")

    train_tok, train_trunc = _tokenize_split(
        raw["train"], tokenizer, text_a, text_b, max_length
    )
    val_tok, val_trunc = _tokenize_split(
        raw[val_key], tokenizer, text_a, text_b, max_length
    )
    num_labels = len(set(raw["train"]["label"]))
    # Report the train-split truncation rate (validation rate is usually similar);
    # both are logged by the caller via this dataclass + val recomputation if needed.
    return NovelTaskData(
        task=task,
        train=train_tok,
        validation=val_tok,
        num_labels=num_labels,
        max_length=max_length,
        truncation_rate=round(max(train_trunc, val_trunc), 6),
    )


# ----------------------------------------------------------------------------
# Stratified subsampling
# ----------------------------------------------------------------------------
def stratified_subsample_indices(labels, budget, seed) -> np.ndarray:
    """Return indices of a label-stratified subset of size ``budget``.

    Guarantees at least one example per present class, so a 16-example few-shot
    subset is never single-class. ``budget == "full"`` returns all indices.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if budget == "full" or budget >= n:
        return np.arange(n)

    rng = np.random.RandomState(seed)
    classes, counts = np.unique(labels, return_counts=True)
    # proportional allocation, floored, then top up to hit the exact budget
    alloc = np.maximum(1, np.floor(budget * counts / n).astype(int))
    while alloc.sum() > budget:  # trim from the largest class
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < budget:  # add to the class with most spare capacity
        spare = counts - alloc
        alloc[np.argmax(spare)] += 1

    picked = []
    for cls, k in zip(classes, alloc):
        cls_idx = np.where(labels == cls)[0]
        picked.append(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def stratified_inner_split(labels, seed, frac=INNER_SPLIT_FRAC):
    """Split indices into (train_idx, selection_idx), stratified by label.

    Used for the inner-split expert-selection protocol: ``selection_idx`` picks
    the expert, the cost stays inside the budget.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train_idx, sel_idx = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0].copy()
        rng.shuffle(cls_idx)
        n_sel = max(1, int(round(frac * len(cls_idx))))
        sel_idx.append(cls_idx[:n_sel])
        train_idx.append(cls_idx[n_sel:])
    return (
        np.concatenate(train_idx) if train_idx else np.array([], dtype=int),
        np.concatenate(sel_idx) if sel_idx else np.array([], dtype=int),
    )


# ----------------------------------------------------------------------------
# Linear probe + metrics
# ----------------------------------------------------------------------------
@dataclass
class ProbeResult:
    mcc: float
    accuracy: float
    degenerate: bool  # True if the probe predicted a single class on val
    n_train: int


def train_probe_and_eval(
    train_features, train_labels, val_features, val_labels
) -> ProbeResult:
    """Fit a fixed-hyperparameter L2 logistic regression and evaluate.

    No early stopping and no probe-dev split: ``PROBE_C`` / ``PROBE_MAX_ITER``
    are protocol constants, identical across every arm/expert/budget, so the
    novel-task validation set never informs model selection.
    """
    clf = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER)
    clf.fit(np.asarray(train_features), np.asarray(train_labels))
    preds = clf.predict(np.asarray(val_features))
    val_labels = np.asarray(val_labels)

    degenerate = len(np.unique(preds)) == 1
    accuracy = float(accuracy_score(val_labels, preds))
    # Degenerate guard: a single-class prediction is not a real score; MCC is
    # undefined / 0, so we set it to 0 explicitly and flag the case.
    mcc = 0.0 if degenerate else float(matthews_corrcoef(val_labels, preds))
    return ProbeResult(
        mcc=round(mcc, 6),
        accuracy=round(accuracy, 6),
        degenerate=degenerate,
        n_train=len(train_labels),
    )


def primary_score(result: ProbeResult, task: str) -> float:
    """Return the pre-registered primary metric for a task."""
    metric = PRIMARY_METRIC[task.lower()]
    return result.mcc if metric == "mcc" else result.accuracy


def majority_baseline(labels) -> float:
    """Majority-class accuracy -- reported alongside BoolQ accuracy."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))


# ----------------------------------------------------------------------------
# Paired delta-over-base statistics
# ----------------------------------------------------------------------------
@dataclass
class PairedDelta:
    """Result of one paired delta-over-base analysis."""

    n_pairs: int
    mean_delta: float
    ci95_low: float
    ci95_high: float
    p_value: float  # Wilcoxon signed-rank (two-sided)
    test: str = "wilcoxon"


def paired_delta_over_base(expert_scores, base_scores) -> PairedDelta:
    """Paired ``expert - base`` analysis.

    ``expert_scores`` and ``base_scores`` are aligned lists: element i is the
    same paired unit (same seed / expert id / budget / subsample). few-shot and
    ``full`` budgets must be passed in separate calls -- never pooled.
    """
    expert = np.asarray(expert_scores, dtype=float)
    base = np.asarray(base_scores, dtype=float)
    if expert.shape != base.shape:
        raise ValueError(
            f"paired arrays must align: {expert.shape} vs {base.shape}"
        )
    deltas = expert - base
    n = len(deltas)
    mean = float(deltas.mean())

    if n < 2:
        return PairedDelta(n, round(mean, 6), mean, mean, math.nan)

    sem = float(deltas.std(ddof=1) / math.sqrt(n))
    half = sem * float(scipy_stats.t.ppf(0.975, df=n - 1))
    if np.allclose(deltas, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(scipy_stats.wilcoxon(deltas).pvalue)
        except ValueError:
            p_value = math.nan
    return PairedDelta(
        n_pairs=n,
        mean_delta=round(mean, 6),
        ci95_low=round(mean - half, 6),
        ci95_high=round(mean + half, 6),
        p_value=p_value,
    )


# ----------------------------------------------------------------------------
# Cache-key helper (used by CU1b for the keyed feature cache)
# ----------------------------------------------------------------------------
def feature_cache_key(metadata: dict) -> str:
    """Hash a feature-cache metadata dict into a stable on-disk filename stem.

    ``metadata`` must contain: model_kind, checkpoint_dir, params_fingerprint,
    arm, seed, task, split, expert_id, routing_mode, tokenizer_name,
    max_length, feature_layer. ``params_fingerprint`` (size+mtime of the
    resolved params file) ensures features extracted from a different
    parameter snapshot are never silently reused. For
    ``model_kind == "base_control"`` the checkpoint/params_fingerprint/seed/
    expert_id/routing_mode fields are the literal string "na".
    """
    required = (
        "model_kind", "checkpoint_dir", "params_fingerprint", "arm", "seed",
        "task", "split", "expert_id", "routing_mode", "tokenizer_name",
        "max_length", "feature_layer",
    )
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(f"feature_cache_key missing fields: {missing}")
    payload = "|".join(f"{k}={metadata[k]}" for k in required)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:16]
    return f"{metadata['task']}_{metadata['split']}_{metadata['model_kind']}_{digest}"


# ============================================================================
# CU1b — checkpoint loading, forced single-expert routing, frozen feature
# extraction, keyed feature cache, and the main() pipeline.
# ============================================================================

FEATURE_CACHE_DIR = ROOT / "scripts" / "cold_start_results" / "_feature_cache"


# ----------------------------------------------------------------------------
# Frozen CLS feature extraction
# ----------------------------------------------------------------------------
def _cls_features_from_backbone(backbone, tokenized_split, tokenizer, device, batch_size):
    """Run ``backbone`` over a tokenized split; return (features, labels) arrays.

    Feature = last-hidden-state CLS token (first position). Frozen — the model
    is never updated; only the linear probe is trained downstream.
    """
    loader = DataLoader(
        tokenized_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    feats, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            lab = batch.get("labels", batch.get("label"))
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ("labels", "label") and isinstance(v, torch.Tensor)
            }
            out = backbone(**inputs)
            cls = out.last_hidden_state[:, 0, :].detach().float().cpu()
            feats.append(cls)
            labels.append(torch.as_tensor(lab))
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def _get_backbone(seq_model):
    """Return the transformer backbone (e.g. RobertaModel) of a sequence model."""
    prefix = getattr(seq_model, "base_model_prefix", None)
    if prefix and hasattr(seq_model, prefix):
        return getattr(seq_model, prefix)
    if hasattr(seq_model, "roberta"):
        return seq_model.roberta
    raise NotImplementedError(
        "cold_start_reuse expects a HF sequence classifier with a named "
        f"transformer backbone; got {type(seq_model).__name__}"
    )


def _cached_features(meta, extract_fn):
    """Return (features, labels), loading from / saving to the keyed cache."""
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_CACHE_DIR / f"{feature_cache_key(meta)}.npz"
    if path.exists():
        with np.load(path) as blob:
            return blob["features"], blob["labels"]
    features, labels = extract_fn()
    # atomic write: a unique temp file + os.replace, so a parallel job sharing
    # this cache key (e.g. base-control features) never reads a partial .npz.
    # The temp name must end in ".npz" -- np.savez auto-appends ".npz" otherwise,
    # which would desync the written file from the os.replace source.
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npz")
    np.savez(tmp, features=features, labels=labels)
    os.replace(tmp, path)
    return features, labels


def extract_expert_features(client, cluster_id, task_data, tokenizer, device,
                            batch_size, cache_meta_base):
    """Extract frozen CLS features for reused expert ``cluster_id`` (train+val)."""
    seq_model = _get_sequence_classifier_model(client.local_model)
    backbone = _get_backbone(seq_model)
    out = {}
    for split_name, split in (("train", task_data.train), ("validation", task_data.validation)):
        meta = dict(cache_meta_base)
        meta.update(
            model_kind="checkpoint_expert", split=split_name,
            expert_id=cluster_id, routing_mode="home_cluster_only",
        )

        def _extract():
            with _forced_cluster_routing(client.local_model, cluster_id):
                return _cls_features_from_backbone(
                    backbone, split, tokenizer, device, batch_size
                )

        out[split_name] = _cached_features(meta, _extract)
    return out


def extract_base_features(base_model, task_data, tokenizer, device, batch_size,
                          cache_meta_base):
    """Extract frozen CLS features from the clean base model (no LoRA)."""
    out = {}
    for split_name, split in (("train", task_data.train), ("validation", task_data.validation)):
        meta = dict(cache_meta_base)
        meta.update(
            model_kind="base_control", split=split_name,
            checkpoint_dir="na", params_fingerprint="na", arm="na", seed="na",
            expert_id="na", routing_mode="na",
        )

        def _extract():
            return _cls_features_from_backbone(
                base_model, split, tokenizer, device, batch_size
            )

        out[split_name] = _cached_features(meta, _extract)
    return out


# ----------------------------------------------------------------------------
# Cold-start probing across budgets / subsamples / experts
# ----------------------------------------------------------------------------
def _budget_subsample_seeds(budget):
    """Subsample seeds for a budget: few-shot gets N_SUBSAMPLES, full gets one."""
    if budget == "full":
        return [0]
    return list(range(N_SUBSAMPLES))


def _probe_score(train_feats, train_labels, val_feats, val_labels, task):
    res = train_probe_and_eval(train_feats, train_labels, val_feats, val_labels)
    return primary_score(res, task), res


def run_cold_start_for_task(task, expert_feats, base_feats, cluster_ids, limit_budgets=None):
    """Run the full budget × subsample × expert cold-start grid for one task.

    expert_feats: {cluster_id: {"train": (X,y), "validation": (X,y)}}
    base_feats:   {"train": (X,y), "validation": (X,y)}
    Returns a nested results dict.
    """
    budgets = [b for b in BUDGETS if (limit_budgets is None or b in limit_budgets)]
    base_tr_X, base_tr_y = base_feats["train"]
    base_va_X, base_va_y = base_feats["validation"]
    val_labels_ref = base_va_y  # validation labels are split-fixed, identical everywhere

    per_budget = {}
    for budget in budgets:
        rows = []
        for sub_seed in _budget_subsample_seeds(budget):
            idx = stratified_subsample_indices(base_tr_y, budget, sub_seed)
            # base probe
            base_score, base_res = _probe_score(
                base_tr_X[idx], base_tr_y[idx], base_va_X, base_va_y, task
            )
            # expert probes (one per reused cluster expert)
            expert_scores = {}
            for c in cluster_ids:
                tr_X, tr_y = expert_feats[c]["train"]
                va_X, va_y = expert_feats[c]["validation"]
                score, _ = _probe_score(tr_X[idx], tr_y[idx], va_X, va_y, task)
                expert_scores[c] = score
            # inner-split expert selection (cost stays inside the budget)
            sel_expert, sel_score = _inner_select(
                expert_feats, cluster_ids, base_tr_y, idx, sub_seed, task
            )
            rows.append({
                "subsample_seed": sub_seed,
                "n_train": int(len(idx)),
                "base_score": base_score,
                "base_degenerate": base_res.degenerate,
                "expert_scores": {str(c): expert_scores[c] for c in cluster_ids},
                "mean_expert_score": float(np.mean(list(expert_scores.values()))),
                "oracle_best_score": float(max(expert_scores.values())),
                "inner_selected_expert": int(sel_expert),
                "inner_selected_score": sel_score,
            })
        per_budget[str(budget)] = {
            "rows": rows,
            "mean_expert_delta": float(np.mean(
                [r["mean_expert_score"] - r["base_score"] for r in rows]
            )),
            "inner_selected_delta": float(np.mean(
                [r["inner_selected_score"] - r["base_score"] for r in rows]
            )),
            "oracle_best_delta": float(np.mean(
                [r["oracle_best_score"] - r["base_score"] for r in rows]
            )),
        }
    return per_budget


def _inner_select(expert_feats, cluster_ids, train_labels_full, budget_idx,
                  sub_seed, task):
    """Inner-split expert selection: pick the best expert on a 30% stratified
    inner split of the budget, then retrain it on the *full* budget and score
    on validation. Returns (selected_cluster_id, validation_score).
    """
    budget_labels = train_labels_full[budget_idx]
    inner_tr_local, inner_sel_local = stratified_inner_split(budget_labels, sub_seed)
    inner_tr = budget_idx[inner_tr_local]
    inner_sel = budget_idx[inner_sel_local]

    best_c, best_inner = cluster_ids[0], -math.inf
    for c in cluster_ids:
        tr_X, tr_y = expert_feats[c]["train"]
        score, _ = _probe_score(
            tr_X[inner_tr], tr_y[inner_tr], tr_X[inner_sel], tr_y[inner_sel], task
        )
        if score > best_inner:
            best_inner, best_c = score, c
    # retrain chosen expert on the full budget, score once on validation
    tr_X, tr_y = expert_feats[best_c]["train"]
    va_X, va_y = expert_feats[best_c]["validation"]
    final_score, _ = _probe_score(
        tr_X[budget_idx], tr_y[budget_idx], va_X, va_y, task
    )
    return best_c, final_score


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="New-client cold-start expert-reuse experiment (one checkpoint)."
    )
    p.add_argument("--checkpoint-dir", required=True,
                   help="Run directory with final_params.pt + training_history.json")
    p.add_argument("--output", default=None, help="Results JSON path")
    p.add_argument("--arm", default="unknown",
                   help="Arm label (baseline_off / shuffled / uniform_both)")
    p.add_argument("--seed", default="unknown", help="Seed label for this checkpoint")
    p.add_argument("--tasks", nargs="+", default=list(NOVEL_TASKS),
                   help="Novel tasks to evaluate (default: cola boolq)")
    p.add_argument("--model-name", default=None,
                   help="Override model_name if not recoverable from the checkpoint")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit-budgets", default=None,
                   help="Debug: comma-separated budget subset, e.g. '16,full'")
    p.add_argument("--limit-experts", type=int, default=None,
                   help="Debug: evaluate only the first N cluster experts")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.checkpoint_dir).resolve()

    history_path = _resolve_training_history_path(run_dir)
    with history_path.open() as f:
        history = json.load(f)
    # Always load latest.json (if present) for run-config metadata: settings
    # such as shared_lora_a / model_name are recovered from the checkpoint args,
    # not just training_history.json. final_params still come from
    # final_params.pt preferentially (see _load_final_params).
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
        raise NotImplementedError("cold_start_reuse supports non-FedRoD checkpoints only")

    cluster_ids = sorted(cluster_metadata)
    if args.limit_experts is not None:
        cluster_ids = cluster_ids[: args.limit_experts]
    # normalize task names once -- load_novel_task lowercases internally, so
    # downstream PRIMARY_METRIC / result keys must use the same casing.
    tasks = [t.lower() for t in args.tasks]
    limit_budgets = None
    if args.limit_budgets is not None:
        limit_budgets = {
            (b if b == "full" else int(b))
            for b in args.limit_budgets.split(",")
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], cache_dir=config["cache_path"]
    )

    print(f"Checkpoint: {run_dir}")
    print(f"Params source: {params_source}")
    print(f"Model: {config['model_name']} | clusters: {cluster_ids}")
    print(f"Tasks: {tasks} | device: {device}")

    # one client model carries every cluster expert; routing selects which.
    client = _build_client(
        client_id=0, task_info=task_info, tokenizer=tokenizer, config=config,
        num_clients=len(task_info), client_to_cluster=client_to_cluster,
    )
    client.load_model()
    client.load_params(_client_params(final_params, 0))
    client.local_model.to(device).eval()

    base_model = AutoModel.from_pretrained(
        config["model_name"], cache_dir=config["cache_path"]
    ).to(device).eval()

    # params_fingerprint: size+mtime of the resolved params file, so cached
    # features are invalidated if the checkpoint snapshot is ever replaced.
    src_path = Path(params_source)
    if src_path.exists():
        st = src_path.stat()
        params_fingerprint = f"{st.st_size}_{int(st.st_mtime)}"
    else:
        params_fingerprint = "unknown"

    cache_meta_base = {
        "checkpoint_dir": str(run_dir), "params_fingerprint": params_fingerprint,
        "arm": args.arm, "seed": args.seed,
        "tokenizer_name": config["model_name"], "feature_layer": "cls_last_hidden",
    }

    results = {
        "checkpoint_dir": str(run_dir), "arm": args.arm, "seed": args.seed,
        "params_source": params_source, "model_name": config["model_name"],
        "cluster_ids": cluster_ids,
        "protocol": {
            "probe_C": PROBE_C, "probe_max_iter": PROBE_MAX_ITER,
            "budgets": list(BUDGETS), "n_subsamples": N_SUBSAMPLES,
            "inner_split_frac": INNER_SPLIT_FRAC,
        },
        "tasks": {},
    }

    for task in tasks:
        print(f"\n=== task: {task} ===")
        task_data = load_novel_task(task, tokenizer)
        meta = dict(cache_meta_base)
        meta.update(task=task, max_length=task_data.max_length)

        expert_feats = {
            c: extract_expert_features(
                client, c, task_data, tokenizer, device, args.batch_size, meta
            )
            for c in cluster_ids
        }
        base_feats = extract_base_features(
            base_model, task_data, tokenizer, device, args.batch_size, meta
        )
        per_budget = run_cold_start_for_task(
            task, expert_feats, base_feats, cluster_ids, limit_budgets
        )
        results["tasks"][task] = {
            "primary_metric": PRIMARY_METRIC[task],
            "num_labels": task_data.num_labels,
            "truncation_rate": task_data.truncation_rate,
            "majority_baseline_val": majority_baseline(base_feats["validation"][1]),
            "per_budget": per_budget,
        }

    _free_client_model(client)
    output_path = Path(args.output) if args.output else run_dir / "cold_start_reuse.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
