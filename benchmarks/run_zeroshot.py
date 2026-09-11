"""run_zeroshot.py — zero-shot workload generalization (context.txt §5).

The Virtual Cell Challenge holds out CELL contexts; we hold out WORKLOAD
contexts (disjoint simulator seeds + extreme demand bands). Answers: did the
surrogate learn OS dynamics, or memorize training workloads?

Scores on held-out episodes: single-step MSE/MAE, rollout drift curves,
hallucination rate, ranking fidelity, calibration. A train-vs-test gap table
distinguishes memorization from learning.

Usage:  python benchmarks/run_zeroshot.py [--full]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._common import ensure_dir, log_manifest, md_table, save_json, timed


def run(full: bool = False) -> dict:
    import numpy as np
    from kernel.policy.schemas import PolicyAction, SchedulerAction
    from virtual_kernel import (
        VKConfig, EnsembleDynamics, TruthOracle, fast_config,
        summarize_oracle_report, spearman_rank, calibration as cal_fn,
        regime_histogram,
    )
    from virtual_kernel.datasets import (
        capture_traces, dataset_hash, to_transitions,
    )

    cfg0 = VKConfig(name="zeroshot-full") if full else fast_config("zeroshot-smoke")
    n_train = 16 if full else 4
    n_test = 12 if full else 3
    steps = 30 if full else 10
    epochs = 200 if full else 40
    train_seeds = list(range(7000, 7000 + n_train))
    test_seeds = list(range(9000, 9000 + n_test))

    from virtual_kernel.datasets import _reseed
    train_eps = _reseed([], train_seeds, steps, cfg0.arms, cfg0.n_cpus, cfg0.dt_s)
    test_eps = _reseed([], test_seeds, steps, cfg0.arms, cfg0.n_cpus, cfg0.dt_s)
    train_ds = to_transitions(train_eps, cfg0.dt_s)
    test_ds = to_transitions(test_eps, cfg0.dt_s)

    ens, t_train = timed(EnsembleDynamics.train, n_members=cfg0.n_members,
                         epochs=epochs, n_cpus=cfg0.n_cpus, dataset=train_ds)
    tr_eval = ens.evaluate(train_ds)
    te_eval = ens.evaluate(test_ds)
    cal = ens.calibration(test_ds)

    # rollout drift on held-out episodes (fixed probe pattern)
    oracle = TruthOracle(seed=0, n_cpus=cfg0.n_cpus, dt_s=cfg0.dt_s)
    probe = [PolicyAction(policy_id="z", scheduler=SchedulerAction(
        target_latency_us=a)) for a in (2000, 6000, 12000, 6000, 2000, 6000)[:6]]
    curves, sq = [], []
    for ep in test_eps:
        rep = oracle.evaluate(ens, ep.steps[0].kir, probe,
                              divergence_threshold=cfg0.divergence_threshold)
        curves.append(rep.rollout_mae_curve)
        sq.append(rep.single_step_mse)
    summary = summarize_oracle_report(float(np.mean(sq)), curves,
                                      cfg0.divergence_threshold)

    # ranking fidelity on held-out start states
    cands = [[2000, 6000], [6000, 6000], [12000, 2000], [2000, 2000]]
    from virtual_kernel.planning import rollout_sequence_reward
    rhos = []
    for ep in test_eps[:4]:
        k0 = ep.steps[0].kir
        pred = [rollout_sequence_reward(ens, k0, s)[0] for s in cands]
        true = [oracle.sequence_reward(k0, s) for s in cands]
        r = spearman_rank(pred, true)
        if r == r:
            rhos.append(r)

    out = {
        "config": cfg0.to_dict(), "config_hash": cfg0.hash,
        "dataset_hash_train": dataset_hash(train_ds),
        "dataset_hash_test": dataset_hash(test_ds),
        "train_regimes": regime_histogram(train_eps),
        "test_regimes": regime_histogram(test_eps),
        "train_mse": round(tr_eval["mse"], 5),
        "test_mse": round(te_eval["mse"], 5),
        "generalization_gap": round(te_eval["mse"] - tr_eval["mse"], 5),
        "rollout": summary,
        "mean_ranking_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
        "calibration": cal,
        "train_s": round(t_train, 2),
    }
    rep_dir = ensure_dir(os.path.join(cfg0.report_dir, "zeroshot"))
    save_json(os.path.join(rep_dir, "results.json"), out)
    md = ("# Zero-shot workload generalization\n\n"
          f"train MSE {out['train_mse']} vs test MSE {out['test_mse']} "
          f"(gap {out['generalization_gap']})\n\n"
          f"rollout: mean divergence step {summary['mean_divergence_step']:.1f}, "
          f"hallucination rate {summary['hallucination_rate']:.2f}\n\n"
          f"ranking spearman {out['mean_ranking_spearman']}, "
          f"calibration corr {cal['uncertainty_error_corr']}\n")
    with open(os.path.join(rep_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        log_manifest("zeroshot", cfg0, out["dataset_hash_train"] + "+" +
                     out["dataset_hash_test"], ens.hashes(),
                     {"gap": out["generalization_gap"], "rollout": summary})
    except Exception as e:
        out["manifest_warning"] = str(e)
    return out


if __name__ == "__main__":
    out = run(full="--full" in sys.argv)
    print({k: out[k] for k in ("train_mse", "test_mse", "generalization_gap",
                               "mean_ranking_spearman")})
    print(out["rollout"])
