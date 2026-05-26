#!/usr/bin/env python
"""Smoke checks for learned cluster affinity.

Runs smoke checks:
  (a) default affinity-off path matches explicit --affinity_mode off
  (b) tau -> infinity learned affinity is uniform over non-home clusters
  (c) learned-affinity checkpoint resume preserves current_alpha
  (d) shuffled mode permutes alpha rows even for identical signatures
  (e)-(o) side-isolated alpha dispatch, migration, and uniform_nonhome checks
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root (scripts/verify/<file>)
TEST_PREFIX = "[verify_learned_affinity]"
HISTORY_VOLATILE_KEY_RE = re.compile(
    r"(_time$|timestamp|runtime$|samples_per_second$|steps_per_second$|model_preparation_time$)",
    re.IGNORECASE,
)
FLOAT_TOL = 1e-7


class SmokeFailure(RuntimeError):
    pass


def _make_tiny_bert_model(model_dir: Path) -> None:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = (
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        + list("abcdefghijklmnopqrstuvwxyz")
        + [str(i) for i in range(10)]
    )
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(model_dir, safe_serialization=True)


def _write_cpu_sitecustomize(shim_dir: Path) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import os
            import json

            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

            import server

            _orig_server_init = server.Server.__init__

            def _cpu_server_init(self, clients_num, device="cuda"):
                if device == "cuda":
                    device = "cpu"
                return _orig_server_init(self, clients_num, device=device)

            server.Server.__init__ = _cpu_server_init

            if os.environ.get("FEDLEASE_CAPTURE_SOFT_MEMBERSHIP"):
                import client

                _orig_client_local_training = client.Client.local_training

                def _capture_client_local_training(self, *args, **kwargs):
                    capture_path = os.environ.get("FEDLEASE_CAPTURE_SOFT_MEMBERSHIP")
                    soft_membership = kwargs.get("soft_membership_for_client")
                    if capture_path and soft_membership is not None:
                        record = {
                            "client_id": int(self.client_id),
                            "task_name": str(getattr(self, "task_name", "")),
                            "soft_membership_for_client": {
                                str(expert_idx): float(weight)
                                for expert_idx, weight in soft_membership.items()
                            },
                        }
                        with open(capture_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, sort_keys=True) + "\\n")
                    return _orig_client_local_training(self, *args, **kwargs)

                client.Client.local_training = _capture_client_local_training
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _subprocess_env(shim_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONHASHSEED"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DISABLED"] = "true"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for key in (
        "SLURMLAB_OUTPUT_DIR",
        "SLURMLAB_RESUME_CKPT",
        "AUTORESEARCH_RESULTS_PATH",
        "MLFLOW_RUN_ID",
        "FEDLEASE_CAPTURE_SOFT_MEMBERSHIP",
    ):
        env.pop(key, None)
    return env


def _base_main_args(model_dir: Path, output_dir: Path, global_rounds: int, warmup_rounds: int) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--model_name",
        str(model_dir),
        "--tasks",
        "sst2",
        "mrpc",
        "--output_dir",
        str(output_dir),
        "--assignment_mode",
        "oracle",
        "--rank",
        "2",
        "--batch_size",
        "8",
        "--train_samples",
        "2",
        "--test_samples",
        "2",
        "--global_rounds",
        str(global_rounds),
        "--warmup_rounds",
        str(warmup_rounds),
        "--local_epochs",
        "1",
        "--soft_membership",
        "task_family",
        "--visa_coeff",
        "0.2",
        "--soft_train_weight",
        "0.20",
        "--seed",
        "42",
        "--universal_expert",
        "--additive_residual",
        "--universal_warmup_rounds",
        "0",
        "--no-save_final_params",
    ]


def _run_main(args: list[str], env: dict[str, str], timeout: int = 300) -> None:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise SmokeFailure(
            f"subprocess failed with exit code {result.returncode}\n"
            f"command: {' '.join(args)}\n"
            f"last output:\n{tail}"
        )


def _find_training_history(output_root: Path) -> Path:
    matches = sorted(output_root.glob("*/proposed_m2/training_history.json"))
    if len(matches) != 1:
        raise SmokeFailure(
            f"expected one training_history.json under {output_root}, found {len(matches)}"
        )
    return matches[0]


def _run_dir_from_history(history_path: Path) -> Path:
    return history_path.parent.parent


def _sanitize_history(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_history(item)
            for key, item in value.items()
            if not HISTORY_VOLATILE_KEY_RE.search(str(key))
        }
    if isinstance(value, list):
        return [_sanitize_history(item) for item in value]
    return value


def _history_hash(history_path: Path) -> str:
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    payload = json.dumps(
        _sanitize_history(history),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _latest_checkpoint(run_dir: Path) -> dict[str, Any]:
    pointer_path = run_dir / "checkpoints" / "latest.json"
    if not pointer_path.exists():
        raise SmokeFailure(f"missing checkpoint pointer: {pointer_path}")
    with pointer_path.open("r", encoding="utf-8") as f:
        pointer = json.load(f)
    checkpoint_path = Path(pointer.get("checkpoint_path") or pointer.get("path"))
    if not checkpoint_path.exists():
        raise SmokeFailure(f"latest checkpoint does not exist: {checkpoint_path}")
    return _torch_load(checkpoint_path)


def _find_round3_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    for path in sorted(checkpoint_dir.glob("*.pt")):
        payload = _torch_load(path)
        progress = payload.get("progress") or {}
        if (
            payload.get("tag") == "round_post_eval"
            and progress.get("phase") == "clustered"
            and int(progress.get("next_round_idx", -1)) == 1
        ):
            return path
    raise SmokeFailure(f"could not find clustered round-3 post-eval checkpoint under {checkpoint_dir}")


def _current_alpha(checkpoint: dict[str, Any]) -> dict[int, dict[int, float]]:
    learned = ((checkpoint.get("fed_state") or {}).get("learned_affinity") or {})
    alpha = learned.get("current_alpha")
    if not alpha:
        raise SmokeFailure("checkpoint has no learned_affinity.current_alpha")
    return {
        int(home): {int(expert): float(weight) for expert, weight in row.items()}
        for home, row in alpha.items()
    }


def _assert_alpha_close(actual: dict[int, dict[int, float]], expected: dict[int, dict[int, float]]) -> None:
    if set(actual) != set(expected):
        raise SmokeFailure(f"alpha home keys differ: {set(actual)} != {set(expected)}")
    for home, actual_row in actual.items():
        expected_row = expected[home]
        if set(actual_row) != set(expected_row):
            raise SmokeFailure(
                f"alpha expert keys differ for home {home}: {set(actual_row)} != {set(expected_row)}"
            )
        for expert, actual_weight in actual_row.items():
            expected_weight = expected_row[expert]
            if abs(actual_weight - expected_weight) > FLOAT_TOL:
                raise SmokeFailure(
                    f"alpha[{home}][{expert}] differs: {actual_weight} vs {expected_weight}"
                )


def _import_project_main():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import main as project_main

        project_main._ensure_project_imports()
    return project_main


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sample_affinity_inputs():
    import torch

    rng = torch.Generator().manual_seed(20260510)
    client_signatures = {
        client_id: torch.randn(8, generator=rng)
        for client_id in range(4)
    }
    lora_client_map = {client_id: [client_id] for client_id in range(4)}
    return client_signatures, lora_client_map


def _refresh_affinity_for_modes(
    client_alpha_mode: str,
    server_alpha_mode: str,
    *,
    base_membership: dict[int, dict[int, float]] | None = None,
    previous_state: dict[str, Any] | None = None,
):
    project_main = _import_project_main()
    client_signatures, lora_client_map = _sample_affinity_inputs()
    with contextlib.redirect_stdout(io.StringIO()):
        return project_main._refresh_learned_affinity_state(
            client_signatures,
            lora_client_map,
            num_clients=4,
            round_number=2,
            affinity_tau=0.5,
            client_alpha_mode=client_alpha_mode,
            server_alpha_mode=server_alpha_mode,
            affinity_shuffle_seed=7,
            base_membership=base_membership,
            log_file=os.devnull,
            previous_state=previous_state,
        )


def _build_test_fed_state(
    learned_affinity: dict[str, Any],
    *,
    client_alpha_mode: str,
    server_alpha_mode: str,
) -> dict[str, Any]:
    project_main = _import_project_main()
    return project_main._build_fed_state(
        aggregated_params=None,
        saved_params=None,
        lora_client_map={0: [0], 1: [1], 2: [2], 3: [3]},
        optimal_n_clusters=4,
        universal_idx=4,
        universal_init_params=None,
        soft_membership=None,
        per_cluster_init_params=None,
        assignment_mode="oracle",
        soft_membership_mode="task_family",
        soft_membership_family_mode="oracle",
        soft_train_weight=0.20,
        soft_floor=0.20,
        uemd_logit_coeff=0.0,
        visa_coeff=0.2,
        visa_conflict_clip=False,
        fedrod_dual_head=False,
        fedrod_alpha_init=2.0,
        fedrod_universal_coeff=1.0,
        fedrod_alpha_coeff=0.0,
        affinity_mode="off",
        client_alpha_mode=client_alpha_mode,
        server_alpha_mode=server_alpha_mode,
        learned_affinity=learned_affinity,
    )


def _normalize_json_int_alpha(alpha: dict[Any, dict[Any, float]] | None) -> dict[int, dict[int, float]] | None:
    if alpha is None:
        return None
    return {
        int(home): {int(expert): float(weight) for expert, weight in row.items()}
        for home, row in alpha.items()
    }


def test_flag_off_bit_identical(model_dir: Path, shim_dir: Path, tmp_root: Path) -> None:
    env = _subprocess_env(shim_dir)
    default_out = tmp_root / "flag_off_default"
    explicit_out = tmp_root / "flag_off_explicit"

    _run_main(_base_main_args(model_dir, default_out, global_rounds=2, warmup_rounds=1), env)

    explicit_args = _base_main_args(model_dir, explicit_out, global_rounds=2, warmup_rounds=1)
    explicit_args += ["--affinity_mode", "off"]
    _run_main(explicit_args, env)

    default_history = _find_training_history(default_out)
    explicit_history = _find_training_history(explicit_out)
    default_hash = _history_hash(default_history)
    explicit_hash = _history_hash(explicit_history)
    if default_hash != explicit_hash:
        raise SmokeFailure(
            "sanitized training_history.json hash mismatch: "
            f"default={default_hash}, explicit_off={explicit_hash}"
        )

    for history_path in (default_history, explicit_history):
        checkpoint = _latest_checkpoint(_run_dir_from_history(history_path))
        fed_state = checkpoint.get("fed_state") or {}
        if "learned_affinity" in fed_state:
            raise SmokeFailure(f"off-path checkpoint unexpectedly wrote learned_affinity: {history_path}")


def test_uniform_tau_limit() -> None:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from main import _build_soft_membership
    from utils import compute_learned_affinity

    k_clusters = 4
    dim = 8
    signatures = {
        cluster_id: torch.nn.functional.one_hot(torch.tensor(cluster_id), num_classes=dim).float()
        for cluster_id in range(k_clusters)
    }
    alpha = compute_learned_affinity(signatures, tau=1e6, mode="full")
    expected_value = 1.0 / (k_clusters - 1)

    for home_cluster, row in alpha.items():
        row_sum = sum(row.values())
        if abs(row_sum - 1.0) > 1e-5:
            raise SmokeFailure(f"alpha row {home_cluster} sums to {row_sum}, not 1.0")
        for expert_idx, weight in row.items():
            if expert_idx == home_cluster:
                raise SmokeFailure(f"alpha row {home_cluster} contains home expert")
            if abs(weight - expected_value) > 1e-5:
                raise SmokeFailure(
                    f"alpha[{home_cluster}][{expert_idx}]={weight}, expected {expected_value}"
                )

    lora_client_map = {cluster_id: [cluster_id] for cluster_id in range(k_clusters)}
    task_names = ["sst2", "mrpc", "qnli", "qqp"]
    task_info = {
        client_id: {"task_name": task_name}
        for client_id, task_name in enumerate(task_names)
    }
    uniform = _build_soft_membership(
        "task_family",
        lora_client_map=lora_client_map,
        task_info=task_info,
        num_clients=k_clusters,
        epsilon=1.0,
        floor=0.20,
        family_mode="all_uniform",
    )

    for home_cluster, row in alpha.items():
        expected_row = uniform[home_cluster]
        if abs(expected_row.get(home_cluster, 0.0)) > 1e-8:
            raise SmokeFailure(f"uniform row {home_cluster} has nonzero home weight")
        for expert_idx, weight in row.items():
            if abs(weight - expected_row[expert_idx]) > 1e-5:
                raise SmokeFailure(
                    f"alpha[{home_cluster}][{expert_idx}]={weight}, "
                    f"uniform={expected_row[expert_idx]}"
                )


def test_shuffled_uniform_signatures_permute_rows() -> None:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from utils import build_learned_affinity_derangement, compute_learned_affinity

    # Use ASYMMETRIC random signatures so pairwise cosines are non-uniform
    # AND the resulting alpha matrix is not circulant. Identical / orthogonal /
    # cyclic signatures all produce α invariant under permutation, which would
    # trivialize the falsification check below.
    k_clusters = 4
    signature_dim = max(8, k_clusters)
    rng = torch.Generator().manual_seed(13)
    signatures = {
        cluster_id: torch.randn(signature_dim, generator=rng)
        for cluster_id in range(k_clusters)
    }
    full = compute_learned_affinity(signatures, tau=0.5, mode="full")
    shuffled = compute_learned_affinity(
        signatures,
        tau=0.5,
        mode="shuffled",
        shuffle_seed=7,
    )
    permutation = build_learned_affinity_derangement(range(k_clusters), shuffle_seed=7)

    if set(shuffled) != set(full):
        raise SmokeFailure(f"shuffled home keys differ: {set(shuffled)} != {set(full)}")
    if any(source == target for source, target in permutation.items()):
        raise SmokeFailure(f"shuffle permutation has a fixed point: {permutation}")

    # Home-exclusion invariant: each shuffled row must not have its own home as expert key.
    for home_cluster, row in shuffled.items():
        if home_cluster in row:
            raise SmokeFailure(
                f"shuffled row {home_cluster} contains home as expert key (broken invariant)"
            )

    # Falsification check: with distinct signatures, shuffled rows must differ from
    # full rows for at least some home cluster. Equality means the permutation
    # had no functional effect.
    if shuffled == full:
        raise SmokeFailure("shuffled alpha unexpectedly matches full alpha")

    # Per the corrected shuffled semantics: shuffled[π(c)][π(k)] == full[c][k].
    for source, target in permutation.items():
        source_row = full[source]
        target_row = shuffled[target]
        # Compare via the permutation of expert keys.
        for expert, expected_weight in source_row.items():
            permuted_expert = permutation[expert]
            actual_weight = target_row.get(permuted_expert)
            if actual_weight is None:
                raise SmokeFailure(
                    f"shuffled[{target}] missing permuted expert {permuted_expert} "
                    f"(from full[{source}][{expert}])"
                )
            if abs(actual_weight - expected_weight) > FLOAT_TOL:
                raise SmokeFailure(
                    f"shuffled[{target}][{permuted_expert}]={actual_weight}, "
                    f"full[{source}][{expert}]={expected_weight}"
                )


def test_resume_round_trip(model_dir: Path, shim_dir: Path, tmp_root: Path) -> None:
    env = _subprocess_env(shim_dir)
    ref_out = tmp_root / "resume_reference"
    resumed_out = tmp_root / "resume_from_round3"

    ref_args = _base_main_args(model_dir, ref_out, global_rounds=4, warmup_rounds=2)
    ref_args += ["--affinity_mode", "full"]
    _run_main(ref_args, env)

    ref_history = _find_training_history(ref_out)
    ref_run_dir = _run_dir_from_history(ref_history)
    round3_checkpoint = _find_round3_checkpoint(ref_run_dir)

    resumed_args = _base_main_args(model_dir, resumed_out, global_rounds=4, warmup_rounds=2)
    resumed_args += ["--affinity_mode", "full", "--resume_from", str(round3_checkpoint)]
    _run_main(resumed_args, env)

    resumed_history = _find_training_history(resumed_out)
    ref_alpha = _current_alpha(_latest_checkpoint(ref_run_dir))
    resumed_alpha = _current_alpha(_latest_checkpoint(_run_dir_from_history(resumed_history)))
    _assert_alpha_close(resumed_alpha, ref_alpha)


def test_client_server_learned_equivalent_to_full() -> None:
    project_main = _import_project_main()
    full_modes = project_main._resolve_alpha_modes("full")
    explicit_modes = project_main._resolve_alpha_modes(
        "off",
        client_alpha="learned",
        server_alpha="learned",
    )
    if full_modes != explicit_modes:
        raise SmokeFailure(f"resolved modes differ: full={full_modes}, explicit={explicit_modes}")

    full_result = _refresh_affinity_for_modes(*full_modes)
    explicit_result = _refresh_affinity_for_modes(*explicit_modes)
    if _canonical_json(full_result) != _canonical_json(explicit_result):
        raise SmokeFailure("--client_alpha learned --server_alpha learned is not byte-identical to full")


def test_client_server_shuffled_equivalent_to_shuffled() -> None:
    project_main = _import_project_main()
    shuffled_modes = project_main._resolve_alpha_modes("shuffled")
    explicit_modes = project_main._resolve_alpha_modes(
        "off",
        client_alpha="shuffled",
        server_alpha="shuffled",
    )
    if shuffled_modes != explicit_modes:
        raise SmokeFailure(
            f"resolved modes differ: shuffled={shuffled_modes}, explicit={explicit_modes}"
        )

    shuffled_result = _refresh_affinity_for_modes(*shuffled_modes)
    explicit_result = _refresh_affinity_for_modes(*explicit_modes)
    for label, result in (("shuffled", shuffled_result), ("explicit", explicit_result)):
        learned_affinity = result[2]
        client_pi = learned_affinity["client_shuffle_permutation"]
        server_pi = learned_affinity["server_shuffle_permutation"]
        if client_pi is not server_pi:
            raise SmokeFailure(f"{label} path did not reuse the same shuffle permutation object")
        if client_pi != server_pi:
            raise SmokeFailure(f"{label} path client/server shuffle permutations differ")
        if learned_affinity["client_alpha"] != learned_affinity["server_alpha"]:
            raise SmokeFailure(f"{label} path client/server shuffled alpha differ")

    if _canonical_json(shuffled_result) != _canonical_json(explicit_result):
        raise SmokeFailure("--client_alpha shuffled --server_alpha shuffled is not byte-identical")


def test_train_only_resolved_predicates() -> None:
    project_main = _import_project_main()
    client_alpha_mode, server_alpha_mode = project_main._resolve_alpha_modes("train_only")
    if (client_alpha_mode, server_alpha_mode) != ("learned", "static"):
        raise SmokeFailure(
            "train_only resolved to "
            f"{(client_alpha_mode, server_alpha_mode)}, expected ('learned', 'static')"
        )

    predicates = project_main._alpha_mode_predicates(client_alpha_mode, server_alpha_mode)
    expected = {
        "client_uses_side_alpha": True,
        "client_uses_dynamic_alpha": True,
        "server_uses_side_alpha": False,
        "server_uses_dynamic_alpha": False,
    }
    for key, expected_value in expected.items():
        if predicates.get(key) is not expected_value:
            raise SmokeFailure(f"{key}={predicates.get(key)}, expected {expected_value}")


def test_uniform_nonhome_k6() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from utils import build_uniform_nonhome_affinity

    cluster_ids = list(range(6))
    alpha = build_uniform_nonhome_affinity(cluster_ids)
    expected_weight = 1.0 / 5.0
    for home in cluster_ids:
        row = alpha.get(home)
        if row is None:
            raise SmokeFailure(f"missing uniform_nonhome row for home {home}")
        if set(row) != set(cluster_ids) - {home}:
            raise SmokeFailure(f"row {home} keys are {set(row)}, expected all non-home clusters")
        for expert, weight in row.items():
            if abs(weight - expected_weight) > FLOAT_TOL:
                raise SmokeFailure(
                    f"alpha[{home}][{expert}]={weight}, expected {expected_weight}"
                )


def test_mixed_learned_shuffled_different_alphas() -> None:
    _, _, learned_affinity = _refresh_affinity_for_modes("learned", "shuffled")
    if learned_affinity["client_shuffle_permutation"] is not None:
        raise SmokeFailure("learned client side unexpectedly has a shuffle permutation")
    if learned_affinity["server_shuffle_permutation"] is None:
        raise SmokeFailure("shuffled server side did not record a shuffle permutation")
    if learned_affinity["client_alpha"] == learned_affinity["server_alpha"]:
        raise SmokeFailure("client learned alpha unexpectedly equals server shuffled alpha")


def test_old_schema_learned_affinity_migration() -> None:
    project_main = _import_project_main()
    current_alpha = {
        "0": {"1": 0.25, "2": 0.75},
        "1": {"0": 0.60, "2": 0.40},
        "2": {"0": 0.35, "1": 0.65},
    }
    expected_alpha = {
        0: {1: 0.25, 2: 0.75},
        1: {0: 0.60, 2: 0.40},
        2: {0: 0.35, 1: 0.65},
    }
    shuffle_permutation = {"0": "1", "1": "2", "2": "0"}
    expected_pi = {0: 1, 1: 2, 2: 0}

    old_shuffled = project_main._normalize_learned_affinity_state(
        {
            "current_alpha": current_alpha,
            "shuffle_permutation": shuffle_permutation,
        }
    )
    for key in ("client_alpha", "server_alpha", "current_alpha"):
        if old_shuffled[key] != expected_alpha:
            raise SmokeFailure(f"old shuffled {key} did not migrate from current_alpha")
    for key in ("client_shuffle_permutation", "server_shuffle_permutation", "shuffle_permutation"):
        if old_shuffled[key] != expected_pi:
            raise SmokeFailure(f"old shuffled {key} did not migrate from shuffle_permutation")

    old_full = project_main._normalize_learned_affinity_state({"current_alpha": current_alpha})
    if old_full["client_alpha"] != expected_alpha or old_full["server_alpha"] != expected_alpha:
        raise SmokeFailure("old full current_alpha did not populate both side alpha keys")
    if (
        old_full["client_shuffle_permutation"] is not None
        or old_full["server_shuffle_permutation"] is not None
        or old_full["shuffle_permutation"] is not None
    ):
        raise SmokeFailure("old full migration unexpectedly populated shuffle permutations")

    if project_main._normalize_learned_affinity_state({}) is not None:
        raise SmokeFailure("empty old learned_affinity state should normalize to None")


def test_uniform_nonhome_k2() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from utils import build_uniform_nonhome_affinity

    alpha = build_uniform_nonhome_affinity([0, 1])
    expected = {0: {1: 1.0}, 1: {0: 1.0}}
    if alpha != expected:
        raise SmokeFailure(f"K=2 uniform_nonhome alpha={alpha}, expected {expected}")


def test_explicit_client_alpha_warning_and_wins() -> None:
    project_main = _import_project_main()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        args = project_main.parse_args(
            ["--affinity_mode", "shuffled", "--client_alpha", "learned"]
        )
    warning = stderr.getvalue()
    if "overrides the value (shuffled) implied" not in warning:
        raise SmokeFailure(f"missing explicit override warning, stderr={warning!r}")
    if args.client_alpha_mode != "learned":
        raise SmokeFailure(f"client_alpha_mode={args.client_alpha_mode}, expected learned")
    if args.server_alpha_mode != "shuffled":
        raise SmokeFailure(f"server_alpha_mode={args.server_alpha_mode}, expected shuffled")


def test_resolved_mode_shuffle_permutation_rules() -> None:
    project_main = _import_project_main()

    with contextlib.redirect_stderr(io.StringIO()):
        m1_args = project_main.parse_args(
            ["--affinity_mode", "shuffled", "--client_alpha", "shuffled"]
        )
    if (m1_args.client_alpha_mode, m1_args.server_alpha_mode) != ("shuffled", "shuffled"):
        raise SmokeFailure(
            "m1 resolved to "
            f"{(m1_args.client_alpha_mode, m1_args.server_alpha_mode)}, expected both shuffled"
        )
    client_pi, server_pi = project_main._build_side_shuffle_permutations(
        m1_args.client_alpha_mode,
        m1_args.server_alpha_mode,
        [0, 1, 2, 3],
        7,
    )
    if client_pi is not server_pi or client_pi != server_pi:
        raise SmokeFailure("m1 did not use the same shuffle permutation for both sides")

    with contextlib.redirect_stderr(io.StringIO()):
        m2_args = project_main.parse_args(
            ["--affinity_mode", "off", "--client_alpha", "shuffled"]
        )
    if (m2_args.client_alpha_mode, m2_args.server_alpha_mode) != ("shuffled", "static"):
        raise SmokeFailure(
            "m2 resolved to "
            f"{(m2_args.client_alpha_mode, m2_args.server_alpha_mode)}, expected client shuffled/server static"
        )
    client_pi, server_pi = project_main._build_side_shuffle_permutations(
        m2_args.client_alpha_mode,
        m2_args.server_alpha_mode,
        [0, 1, 2, 3],
        7,
    )
    if client_pi is None:
        raise SmokeFailure("m2 client shuffle permutation is None")
    if server_pi is not None:
        raise SmokeFailure(f"m2 server shuffle permutation={server_pi}, expected None")


def test_fed_state_writes_new_keys_and_legacy_aliases() -> None:
    _, _, learned_affinity = _refresh_affinity_for_modes("shuffled", "shuffled")
    fed_state = _build_test_fed_state(
        learned_affinity,
        client_alpha_mode="shuffled",
        server_alpha_mode="shuffled",
    )
    saved = fed_state.get("learned_affinity") or {}
    required_keys = {
        "client_alpha",
        "server_alpha",
        "client_shuffle_permutation",
        "server_shuffle_permutation",
        "current_alpha",
        "shuffle_permutation",
    }
    missing = required_keys - set(saved)
    if missing:
        raise SmokeFailure(f"saved learned_affinity missing keys: {sorted(missing)}")
    if saved["current_alpha"] != saved["client_alpha"]:
        raise SmokeFailure("legacy current_alpha does not mirror client_alpha")
    if saved["shuffle_permutation"] != saved["client_shuffle_permutation"]:
        raise SmokeFailure("legacy shuffle_permutation does not mirror client_shuffle_permutation")


def test_uniform_nonhome_client_routing(model_dir: Path, shim_dir: Path, tmp_root: Path) -> None:
    env = _subprocess_env(shim_dir)
    capture_path = tmp_root / "uniform_nonhome_capture.jsonl"
    env["FEDLEASE_CAPTURE_SOFT_MEMBERSHIP"] = str(capture_path)
    output_dir = tmp_root / "uniform_nonhome_route"

    args = _base_main_args(model_dir, output_dir, global_rounds=2, warmup_rounds=1)
    args += ["--client_alpha", "uniform_nonhome", "--server_alpha", "static"]
    _run_main(args, env)

    if not capture_path.exists():
        raise SmokeFailure("uniform_nonhome routing capture file was not written")
    records = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise SmokeFailure("uniform_nonhome routing capture file is empty")

    latest_by_client = {int(record["client_id"]): record for record in records}
    for client_id, nonhome in ((0, 1), (1, 0)):
        record = latest_by_client.get(client_id)
        if record is None:
            raise SmokeFailure(f"missing captured clustered training record for client {client_id}")
        row = {
            int(expert): float(weight)
            for expert, weight in record["soft_membership_for_client"].items()
        }
        routed_weight = row.get(nonhome)
        if routed_weight is None:
            raise SmokeFailure(f"client {client_id} routed row missing non-home expert {nonhome}: {row}")
        if abs(routed_weight - 1.0) > FLOAT_TOL:
            raise SmokeFailure(
                f"client {client_id} routed non-home weight {routed_weight}, expected 1.0"
            )
        if abs(routed_weight - 0.20) <= FLOAT_TOL:
            raise SmokeFailure(f"client {client_id} used static task_family weight instead of uniform_nonhome")

    history_path = _find_training_history(output_dir)
    checkpoint = _latest_checkpoint(_run_dir_from_history(history_path))
    fed_state = checkpoint.get("fed_state") or {}
    if fed_state.get("client_alpha_mode") != "uniform_nonhome":
        raise SmokeFailure(f"saved client_alpha_mode={fed_state.get('client_alpha_mode')}")
    if fed_state.get("server_alpha_mode") != "static":
        raise SmokeFailure(f"saved server_alpha_mode={fed_state.get('server_alpha_mode')}")

    learned = fed_state.get("learned_affinity") or {}
    actual_alpha = _normalize_json_int_alpha(learned.get("client_alpha"))
    expected_alpha = {0: {1: 1.0}, 1: {0: 1.0}}
    if actual_alpha != expected_alpha:
        raise SmokeFailure(f"saved client_alpha={actual_alpha}, expected {expected_alpha}")
    if learned.get("server_alpha") is not None:
        raise SmokeFailure(f"saved server_alpha={learned.get('server_alpha')}, expected None")


def _run_test(label: str, description: str, func) -> bool:
    print(f"{TEST_PREFIX} test ({label}) {description} ...", flush=True)
    try:
        func()
    except Exception as exc:  # noqa: BLE001 - smoke driver reports any failure.
        print(f"{TEST_PREFIX} test ({label}) {description} ... FAIL: {exc}", flush=True)
        return False
    print(f"{TEST_PREFIX} test ({label}) {description} ... PASS", flush=True)
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fedlease_learned_affinity_") as tmp:
        tmp_root = Path(tmp)
        model_dir = tmp_root / "tiny_bert_model"
        shim_dir = tmp_root / "cpu_shim"
        _make_tiny_bert_model(model_dir)
        _write_cpu_sitecustomize(shim_dir)

        tests = [
            (
                "a",
                "flag-off bit-identical",
                lambda: test_flag_off_bit_identical(model_dir, shim_dir, tmp_root),
            ),
            ("b", "uniform-alpha limit", test_uniform_tau_limit),
            (
                "c",
                "state save/load round-trip",
                lambda: test_resume_round_trip(model_dir, shim_dir, tmp_root),
            ),
            (
                "d",
                "shuffled uniform signatures",
                test_shuffled_uniform_signatures_permute_rows,
            ),
            (
                "e",
                "explicit learned equals full",
                test_client_server_learned_equivalent_to_full,
            ),
            (
                "f",
                "explicit shuffled equals shuffled",
                test_client_server_shuffled_equivalent_to_shuffled,
            ),
            (
                "g",
                "train_only resolved predicates",
                test_train_only_resolved_predicates,
            ),
            (
                "h",
                "uniform_nonhome K=6",
                test_uniform_nonhome_k6,
            ),
            (
                "i",
                "mixed learned/shuffled alphas differ",
                test_mixed_learned_shuffled_different_alphas,
            ),
            (
                "j",
                "old-schema learned_affinity migration",
                test_old_schema_learned_affinity_migration,
            ),
            (
                "k",
                "uniform_nonhome K=2",
                test_uniform_nonhome_k2,
            ),
            (
                "l",
                "explicit override warning",
                test_explicit_client_alpha_warning_and_wins,
            ),
            (
                "m",
                "resolved shuffle permutation rules",
                test_resolved_mode_shuffle_permutation_rules,
            ),
            (
                "n",
                "fed_state legacy aliases",
                test_fed_state_writes_new_keys_and_legacy_aliases,
            ),
            (
                "o",
                "uniform_nonhome client routing",
                lambda: test_uniform_nonhome_client_routing(model_dir, shim_dir, tmp_root),
            ),
        ]
        ok = True
        for test in tests:
            ok = _run_test(*test) and ok
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
