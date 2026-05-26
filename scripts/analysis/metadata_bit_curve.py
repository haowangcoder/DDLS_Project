"""
Metadata-Bit Fairness Curve — post-hoc analysis on incumbent cross_eval matrix.

Reframe: compare federated LoRA routing policies on a (B, A) plane where
  B = I(T; S) in bits  — task→selection mutual information ("metadata leak")
  A = expected accuracy under that policy

No GPU needed: routing simulation runs over the K×K cross_eval matrix.
"""

import json
import math
from pathlib import Path

import numpy as np

OUTPUT_ROOT = Path("/storage/homefs/hw24w089/lab/FedLEASE/output")
SEEDS = [42, 44]  # seed 43 uses 4-cluster setup, skip
EXCLUDE = {"mnli"}  # mnli row incomplete in incumbent runs

FAMILIES = {
    "sst2": "sentiment",
    "qnli": "NLI",
    "rte": "NLI",
    "mrpc": "paraphrase",
    "qqp": "paraphrase",
}


def load_matrix(path: Path, exclude=EXCLUDE):
    d = json.loads(path.read_text())
    m = d["cluster_task_matrix"]
    clusters = sorted([c for c in m if c not in exclude])
    tasks = sorted({t for c in clusters for t in m[c] if t not in exclude})
    M = np.full((len(clusters), len(tasks)), np.nan)
    for i, c in enumerate(clusters):
        for j, t in enumerate(tasks):
            if t in m[c]:
                M[i, j] = m[c][t]
    return clusters, tasks, M


def acc_under_policy(pi, M):
    """pi[t][s] = P(select cluster s | input task t). Returns expected accuracy."""
    K_clu, K_tsk = M.shape
    a = 0.0
    for j in range(K_tsk):
        a += sum(pi[j][i] * M[i, j] for i in range(K_clu)) / K_tsk
    return float(a)


def mutual_info_bits(pi, K_clu):
    """I(T; S) for policy pi[t][s] under uniform P(t)."""
    K_tsk = len(pi)
    P_s = np.zeros(K_clu)
    for j in range(K_tsk):
        for i in range(K_clu):
            P_s[i] += pi[j][i] / K_tsk
    I = 0.0
    for j in range(K_tsk):
        for i in range(K_clu):
            p = pi[j][i]
            if p > 0 and P_s[i] > 0:
                I += (1 / K_tsk) * p * math.log2(p / P_s[i])
    return I


def policy_uniform(K_clu, K_tsk):
    return [[1.0 / K_clu] * K_clu for _ in range(K_tsk)]


def policy_oracle(clusters, tasks):
    K_clu, K_tsk = len(clusters), len(tasks)
    pi = [[0.0] * K_clu for _ in range(K_tsk)]
    for j, t in enumerate(tasks):
        if t in clusters:
            pi[j][clusters.index(t)] = 1.0
        else:
            for i in range(K_clu):
                pi[j][i] = 1.0 / K_clu
    return pi


def policy_family(clusters, tasks, families):
    fam_to_clusters = {}
    for t, f in families.items():
        if t in clusters:
            fam_to_clusters.setdefault(f, []).append(t)
    K_clu, K_tsk = len(clusters), len(tasks)
    pi = [[0.0] * K_clu for _ in range(K_tsk)]
    for j, t in enumerate(tasks):
        f = families[t]
        member = fam_to_clusters[f]
        for c in member:
            pi[j][clusters.index(c)] = 1.0 / len(member)
    return pi


def policy_mixed(clusters, tasks, alpha):
    K_clu, K_tsk = len(clusters), len(tasks)
    pi = [[(1.0 - alpha) / K_clu] * K_clu for _ in range(K_tsk)]
    for j, t in enumerate(tasks):
        if t in clusters:
            pi[j][clusters.index(t)] += alpha
    return pi


def policy_best_constant(M):
    K_clu, K_tsk = M.shape
    row_means = [float(np.nanmean(M[i, :])) for i in range(K_clu)]
    best = int(np.argmax(row_means))
    pi = [[0.0] * K_clu for _ in range(K_tsk)]
    for j in range(K_tsk):
        pi[j][best] = 1.0
    return pi, best, row_means[best]


def evaluate_seed(path: Path):
    clusters, tasks, M = load_matrix(path)
    K_clu = len(clusters)
    K_tsk = len(tasks)

    rows = []

    # B=0 endpoints
    pi_u = policy_uniform(K_clu, K_tsk)
    rows.append(("uniform random", mutual_info_bits(pi_u, K_clu), acc_under_policy(pi_u, M)))

    pi_bc, best_idx, _ = policy_best_constant(M)
    rows.append((f"best constant ({clusters[best_idx]})", mutual_info_bits(pi_bc, K_clu), acc_under_policy(pi_bc, M)))

    # mixed curve
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        pi_m = policy_mixed(clusters, tasks, alpha)
        rows.append((f"mixed α={alpha}", mutual_info_bits(pi_m, K_clu), acc_under_policy(pi_m, M)))

    # family routing
    pi_f = policy_family(clusters, tasks, FAMILIES)
    rows.append(("family-level", mutual_info_bits(pi_f, K_clu), acc_under_policy(pi_f, M)))

    # oracle
    pi_o = policy_oracle(clusters, tasks)
    rows.append(("oracle (per-task)", mutual_info_bits(pi_o, K_clu), acc_under_policy(pi_o, M)))

    return clusters, tasks, M, rows


def main():
    print("=" * 70)
    print("Metadata-Bit Fairness Curve — incumbent (18-client, 6-cluster oracle)")
    print(f"Excluded from analysis: {EXCLUDE} (incomplete cross_eval row)")
    print("=" * 70)

    all_rows = []
    for seed in SEEDS:
        path = OUTPUT_ROOT / f"roberta-large_multi_task_federated_18_lr1e-03_seed{seed}/cross_eval_results.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        clusters, tasks, M, rows = evaluate_seed(path)
        print(f"\n--- seed {seed} ---")
        print(f"clusters {clusters}, tasks {tasks}")
        print(f"matrix shape {M.shape}, in_dist diag = {np.nanmean(np.diag(M)):.2f}")
        print(f"{'policy':<25}{'B (bits)':>10}{'A (acc)':>10}")
        for name, B, A in rows:
            print(f"{name:<25}{B:>10.3f}{A:>10.2f}")
        all_rows.append((seed, rows))

    # cross-seed mean
    if len(all_rows) >= 2:
        print("\n--- cross-seed mean ---")
        names = [r[0] for r in all_rows[0][1]]
        Bs_per_policy = {n: [] for n in names}
        As_per_policy = {n: [] for n in names}
        for seed, rows in all_rows:
            for name, B, A in rows:
                Bs_per_policy[name].append(B)
                As_per_policy[name].append(A)
        print(f"{'policy':<25}{'B mean':>10}{'A mean':>10}{'A std':>8}")
        for n in names:
            Bm = np.mean(Bs_per_policy[n])
            Am = np.mean(As_per_policy[n])
            As = np.std(As_per_policy[n])
            print(f"{n:<25}{Bm:>10.3f}{Am:>10.2f}{As:>8.2f}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        markers = {42: "o", 44: "s"}
        colors_per_label = {}
        for seed, rows in all_rows:
            xs = [r[1] for r in rows]
            ys = [r[2] for r in rows]
            labels = [r[0] for r in rows]
            ax.plot(xs, ys, "-", alpha=0.4, color="gray")
            for x, y, lab in zip(xs, ys, labels):
                ax.scatter(x, y, marker=markers[seed], s=60, label=f"{lab} (s{seed})" if seed == SEEDS[0] else None)
        ax.set_xlabel("B = I(T; S)  [bits of task-family metadata leaked to routing]")
        ax.set_ylabel("A = expected accuracy")
        ax.set_title("Metadata-Bit Fairness Curve (incumbent FedLEASE 6-cluster, 5-task subset)")
        ax.grid(True, alpha=0.3)
        # annotate B=0 and B=log2(K) reference lines
        K = 5
        ax.axvline(0, color="C0", linestyle="--", alpha=0.3, label=f"B=0 (task-blind)")
        ax.axvline(math.log2(K), color="C2", linestyle="--", alpha=0.3, label=f"B=log₂({K})={math.log2(K):.2f} (oracle)")
        ax.axvline(math.log2(3), color="C1", linestyle="--", alpha=0.3, label=f"B=log₂(3)={math.log2(3):.2f} (family)")
        ax.legend(fontsize=8, loc="lower right")
        out = OUTPUT_ROOT / "figures" / "metadata_bit_curve.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"\nplot saved -> {out}")
    except ImportError as e:
        print(f"\n[skip plot] matplotlib unavailable: {e}")


if __name__ == "__main__":
    main()
