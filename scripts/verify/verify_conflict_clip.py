#!/usr/bin/env python
"""Smoke checks for conflict-clipped visa FedAvg aggregation."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server import Server  # noqa: E402


def _aggregate(params, visa_conflict_clip):
    server = Server(clients_num=len(params), device="cpu")
    return server.aggregation(
        route_aggregation=True,
        params=params,
        lora_client_map={0: [0], 1: [1]},
        universal_idx=None,
        soft_membership={
            0: {0: 1.0},
            1: {0: 1.0},
        },
        visa_conflict_clip=visa_conflict_clip,
    )


def verify_antihome_lora_b_is_clipped():
    home = torch.tensor([1.0, 0.0])
    visa = torch.tensor([-1.0, 1.0])
    params = [
        {
            "lora_A0.weight": home.clone(),
            "lora_B0.weight": home.clone(),
        },
        {
            "lora_A0.weight": visa.clone(),
            "lora_B0.weight": visa.clone(),
        },
    ]

    unclipped = _aggregate(params, visa_conflict_clip=False)[0]["lora_B0.weight"]
    clipped = _aggregate(params, visa_conflict_clip=True)[0]["lora_B0.weight"]

    expected_unclipped = torch.tensor([0.0, 0.5])
    expected_clipped = torch.tensor([0.5, 0.5])
    assert torch.allclose(unclipped, expected_unclipped), (unclipped, expected_unclipped)
    assert torch.allclose(clipped, expected_clipped), (clipped, expected_clipped)

    recovered_safe_visa = 2.0 * clipped - home
    assert recovered_safe_visa.dot(home).abs() < 1e-6, recovered_safe_visa
    print("PASS CU10.a: anti-home lora_B component is clipped to orthogonal")


def verify_positive_lora_b_is_unchanged():
    home = torch.tensor([1.0, 0.0])
    visa = torch.tensor([1.0, 1.0])
    params = [
        {"lora_B0.weight": home},
        {"lora_B0.weight": visa},
    ]

    unclipped = _aggregate(params, visa_conflict_clip=False)[0]["lora_B0.weight"]
    clipped = _aggregate(params, visa_conflict_clip=True)[0]["lora_B0.weight"]
    assert torch.allclose(clipped, unclipped), (clipped, unclipped)
    print("PASS CU10.b: aligned lora_B visa update is unchanged")


def verify_lora_a_stays_soft_average():
    home = torch.tensor([1.0, 0.0])
    visa = torch.tensor([-1.0, 1.0])
    params = [
        {"lora_A0.weight": home, "lora_B0.weight": home},
        {"lora_A0.weight": visa, "lora_B0.weight": visa},
    ]

    clipped = _aggregate(params, visa_conflict_clip=True)[0]["lora_A0.weight"]
    expected = torch.tensor([0.0, 0.5])
    assert torch.allclose(clipped, expected), (clipped, expected)
    print("PASS CU10.c: lora_A remains the existing soft-membership average")


def main():
    print("=== verify_conflict_clip ===\n")
    verify_antihome_lora_b_is_clipped()
    verify_positive_lora_b_is_unchanged()
    verify_lora_a_stays_soft_average()
    print("\nAll checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
