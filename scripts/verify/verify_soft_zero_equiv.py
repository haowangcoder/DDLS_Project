#!/usr/bin/env python
"""Sanity tests for FedLEASE-Soft (soft cluster membership).

Direct unit-test-style checks on builders and helpers. No federated training
loop. Verifies:
  V1 — `--soft_membership=none` preserves legacy code paths
  V2 — soft-membership math is correct (sums-to-one, ε=0 one-hot, edge cases)
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def verify_soft_membership_builder():
    from main import _build_soft_membership

    # mode='none' returns None (preserves legacy path)
    res = _build_soft_membership(
        "none",
        lora_client_map={0: [0, 1, 2]},
        task_info={cid: {"task_name": "sst2"} for cid in range(3)},
        num_clients=3, epsilon=0.20, floor=0.20,
    )
    assert res is None, f"mode=none must return None, got {res!r}"
    print("PASS V1.a: mode='none' returns None")

    # 6-task GLUE setup: weights sum to 1.0 per client
    lora_map = {i: list(range(3 * i, 3 * (i + 1))) for i in range(6)}
    task_info = {}
    for cid, task in zip(
        range(18),
        ["sst2"] * 3 + ["qnli"] * 3 + ["mrpc"] * 3 + ["qqp"] * 3 + ["mnli"] * 3 + ["rte"] * 3,
    ):
        task_info[cid] = {"task_name": task}

    res = _build_soft_membership(
        "task_family", lora_map, task_info, num_clients=18,
        epsilon=0.20, floor=0.20,
    )
    assert res is not None
    for cid, weights in res.items():
        s = sum(weights.values())
        assert abs(s - 1.0) < 1e-5, f"client {cid}: weights sum to {s}, weights={weights}"
    print(f"PASS V2.a: 6-task task_family weights sum to 1.0 across {len(res)} clients")

    # ε=0: one-hot at oracle cluster
    res0 = _build_soft_membership(
        "task_family", lora_map, task_info, num_clients=18,
        epsilon=0.0, floor=0.20,
    )
    for cid, weights in res0.items():
        oracle = next(k for k, v in lora_map.items() if cid in v)
        assert weights[oracle] == 1.0, f"client {cid}: oracle weight {weights[oracle]} != 1.0"
        for k, w in weights.items():
            if k != oracle:
                assert w < 1e-9, f"client {cid}: non-oracle cluster {k} has weight {w}"
    print("PASS V2.b: ε=0 produces one-hot at oracle for all 18 clients")

    # Edge case: all-NLI (no out-of-family clusters)
    nli_map = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    nli_task_info = {
        cid: {"task_name": t}
        for cid, t in zip(range(9), ["qnli"] * 3 + ["mnli"] * 3 + ["rte"] * 3)
    }
    res_nli = _build_soft_membership(
        "task_family", nli_map, nli_task_info, num_clients=9,
        epsilon=0.20, floor=0.20,
    )
    for cid, weights in res_nli.items():
        s = sum(weights.values())
        assert abs(s - 1.0) < 1e-5, f"all-NLI client {cid}: weights sum to {s}"
    print(f"PASS V2.c: all-NLI edge case (no out-of-family) sums to 1.0 across {len(res_nli)} clients")

    # Edge case: K=1 singleton
    sst_map = {0: [0, 1, 2]}
    sst_task_info = {cid: {"task_name": "sst2"} for cid in range(3)}
    res_sst = _build_soft_membership(
        "task_family", sst_map, sst_task_info, num_clients=3,
        epsilon=0.20, floor=0.20,
    )
    for cid, weights in res_sst.items():
        s = sum(weights.values())
        assert abs(s - 1.0) < 1e-5, f"K=1 client {cid}: weights={weights}"
        assert weights == {0: 1.0}
    print("PASS V2.d: K=1 singleton edge case → home gets 1.0")


def verify_warm_start_legacy_branch():
    from main import _build_clustered_warm_start

    class MockClient:
        idx = 2
        universal_idx = None

    client_params = {
        "lora_A0.weight": torch.randn(4, 8),
        "lora_B0.weight": torch.randn(16, 4),
        "classifier.weight": torch.randn(2, 16),
    }

    # Legacy path: per_cluster_init_params=None
    warmed = _build_clustered_warm_start(MockClient(), client_params, per_cluster_init_params=None)
    assert "lora_A2.weight" in warmed, "Home slot lora_A2 missing in legacy path"
    assert "lora_B2.weight" in warmed, "Home slot lora_B2 missing in legacy path"
    assert "lora_A0.weight" not in warmed, "lora_A0 should be renamed to lora_A2"
    assert "classifier.weight" in warmed
    # Non-home slots NOT populated
    for k in (0, 1, 3, 4, 5):
        assert f"lora_A{k}.weight" not in warmed, f"Legacy path should not populate non-home lora_A{k}"
    print("PASS V1.b: legacy warm start (per_cluster_init=None) only renames home slot")


def verify_warm_start_soft_branch():
    from main import _build_clustered_warm_start

    class MockClient:
        idx = 2
        universal_idx = None

    client_params = {
        "lora_A0.weight": torch.randn(4, 8),
        "lora_B0.weight": torch.randn(16, 4),
        "classifier.weight": torch.randn(2, 16),
    }

    pci = {
        0: {"lora_A0.weight": torch.zeros(4, 8), "lora_B0.weight": torch.zeros(16, 4)},
        1: {"lora_A0.weight": torch.ones(4, 8), "lora_B0.weight": torch.ones(16, 4)},
        2: {"lora_A0.weight": torch.full((4, 8), 2.0), "lora_B0.weight": torch.full((16, 4), 2.0)},
    }
    warmed = _build_clustered_warm_start(MockClient(), client_params, per_cluster_init_params=pci)
    for k in (0, 1, 2):
        assert f"lora_A{k}.weight" in warmed, f"Soft warm start missing lora_A{k}"
        assert f"lora_B{k}.weight" in warmed, f"Soft warm start missing lora_B{k}"
    assert torch.equal(warmed["lora_A0.weight"], torch.zeros(4, 8))
    assert torch.equal(warmed["lora_A1.weight"], torch.ones(4, 8))
    assert torch.equal(warmed["lora_A2.weight"], torch.full((4, 8), 2.0))
    assert "classifier.weight" in warmed
    print("PASS V2.e: soft warm start (per_cluster_init populated) populates all K slots from cluster avg")


def verify_trainable_experts_signature():
    import client as client_module

    res_hard = client_module._build_trainable_experts(2, 5)
    assert res_hard == {2, 5}, f"Hard path: expected {{2,5}}, got {res_hard}"
    print(f"PASS V1.c: _build_trainable_experts hard path → {res_hard}")

    res_soft = client_module._build_trainable_experts(
        2, 5,
        soft_membership_for_client={0: 0.10, 2: 0.70, 4: 0.20},
    )
    assert res_soft == {0, 2, 4, 5}, f"Soft path: expected {{0,2,4,5}}, got {res_soft}"
    print(f"PASS V2.f: _build_trainable_experts soft path → {res_soft}")


def verify_aggregation_signature():
    import inspect
    from server import Server

    sig = inspect.signature(Server.aggregation)
    assert "soft_membership" in sig.parameters, f"Server.aggregation missing soft_membership: {sig}"
    print("PASS V1.d: Server.aggregation has soft_membership=None parameter")


def main():
    print("=== verify_soft_zero_equiv ===\n")
    verify_soft_membership_builder()
    print()
    verify_warm_start_legacy_branch()
    verify_warm_start_soft_branch()
    print()
    verify_trainable_experts_signature()
    verify_aggregation_signature()
    print("\nAll checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
