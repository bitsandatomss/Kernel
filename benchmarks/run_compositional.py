"""run_compositional.py — compositional / sequential interventions
(context.txt §7: the biggest research opportunity).

  T1  sequences vs greedy: does joint sequence search beat greedy one-step
      selection (evidence for non-myopic dynamics)?
  T2  reversal: which intervention reverses another? From a perturbed state,
      search the action that best returns toward the start state.
  T3  pathway walk: sequentially drive a saturated workload toward a
      low-latency state; report trajectory found virtually vs oracle truth.

Usage:  python benchmarks/run_compositional.py [--full]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._common import ensure_dir, log_manifest, md_table, save_json, timed


def run(full: bool = False) -> dict:
    import itertools
    import numpy as np
    from kernel.policy.schemas import PolicyAction, SchedulerAction
    from kernel.simulator.env import KernelSimulator
    from virtual_kernel import (
        VKConfig, EnsembleDynamics, TruthOracle, VirtualKernel, fast_config,
        mcts_search, virtual_screen,
    )
    from virtual_kernel.datasets import capture_traces, dataset_hash, to_transitions
    from virtual_kernel.dynamics import encode_kir
    from virtual_kernel.planning import rollout_sequence_reward

    cfg = VKConfig(name="compositional-full") if full else fast_config("compositional-smoke")
    horizon = 3 if full else 2
    epochs = 150 if full else cfg.epochs
    eps = capture_traces(cfg.n_episodes, cfg.steps_per_episode, base_seed=cfg.seed,
                         arms=cfg.arms, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)
    ds = to_transitions(eps, cfg.dt_s)
    ens = EnsembleDynamics.train(n_members=cfg.n_members, epochs=epochs,
                                 n_cpus=cfg.n_cpus, dataset=ds)
    oracle = TruthOracle(seed=11, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)
    kir0 = KernelSimulator(seed=11, n_cpus=cfg.n_cpus).current_kir()

    # T1: joint sequence search vs greedy one-step
    def greedy(h):
        kir, seq, tot = kir0.model_copy(deep=True), [], 0.0
        for _ in range(h):
            best = max(
                ((a, rollout_sequence_reward(ens, kir, [a])[0]) for a in cfg.arms),
                key=lambda t: t[1])
            seq.append(best[0])
            tot += best[1]
            kir = ens.predict_kir(
                kir, PolicyAction(policy_id="g", scheduler=SchedulerAction(
                    target_latency_us=best[0])), dt_s=cfg.dt_s)
        return seq

    joint = virtual_screen(ens, kir0, horizon=horizon,
                           beam_width=cfg.beam_width, dt_s=cfg.dt_s)[0].latencies
    gseq = greedy(horizon)
    t1 = {"joint_virtual": joint, "greedy_virtual": gseq,
          "joint_oracle": round(oracle.sequence_reward(kir0, joint), 4),
          "greedy_oracle": round(oracle.sequence_reward(kir0, gseq), 4)}

    # T2: reversal — perturb then return
    perturb = [24000] * horizon
    kir_p, _ = rollout_sequence_reward(ens, kir0, perturb)  # virtual perturbed
    # exact perturbed state:
    oracle.reset(kir0)
    kp = kir0
    for lat in perturb:
        kp = oracle.step(PolicyAction(policy_id="p", scheduler=SchedulerAction(
            target_latency_us=lat)))
    best_rev, best_d = None, float("inf")
    for seq in itertools.product(cfg.arms, repeat=horizon):
        r, kend = rollout_sequence_reward(ens, kp, list(seq))
        d = float(np.abs(encode_kir(kend) - encode_kir(kir0)).mean())
        if d < best_d:
            best_d, best_rev = d, list(seq)
    # oracle truth for the chosen reversal
    oracle.reset(kp)
    ko = kp
    for lat in best_rev:
        ko = oracle.step(PolicyAction(policy_id="r", scheduler=SchedulerAction(
            target_latency_us=lat)))
    t2 = {"perturbation": perturb, "reversal_virtual": best_rev,
          "virtual_return_dist": round(best_d, 5),
          "oracle_return_dist": round(
              float(np.abs(encode_kir(ko) - encode_kir(kir0)).mean()), 5)}

    # T3: pathway walk under MCTS toward low latency from saturation
    sat = KernelSimulator(seed=97, n_cpus=cfg.n_cpus).current_kir()
    m = mcts_search(ens, sat, horizon=horizon, sims=150 if full else 40,
                    seed=cfg.seed, arms=tuple(cfg.arms), dt_s=cfg.dt_s)
    t3 = {"mcts_path": m.best_latencies, "mcts_value": round(m.value, 4),
          "oracle_reward": round(oracle.sequence_reward(sat, m.best_latencies), 4)}

    out = {"config": cfg.to_dict(), "config_hash": cfg.hash,
           "T1_joint_vs_greedy": t1, "T2_reversal": t2, "T3_pathway": t3}
    rep_dir = ensure_dir(os.path.join(cfg.report_dir, "compositional"))
    save_json(os.path.join(rep_dir, "results.json"), out)
    md = ("# Compositional interventions\n\n"
          f"T1 joint {t1['joint_virtual']} oracle={t1['joint_oracle']} vs "
          f"greedy {t1['greedy_virtual']} oracle={t1['greedy_oracle']}\n\n"
          f"T2 reversal {t2['reversal_virtual']}: virtual return-dist "
          f"{t2['virtual_return_dist']}, oracle return-dist {t2['oracle_return_dist']}\n\n"
          f"T3 pathway {t3['mcts_path']} value={t3['mcts_value']} "
          f"oracle={t3['oracle_reward']}\n")
    with open(os.path.join(rep_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        log_manifest("compositional", cfg, dataset_hash(ds), ens.hashes(),
                     {"T1": t1, "T2": t2, "T3": t3})
    except Exception as e:
        out["manifest_warning"] = str(e)
    return out


if __name__ == "__main__":
    out = run(full="--full" in sys.argv)
    print(out["T1_joint_vs_greedy"])
    print(out["T2_reversal"])
    print(out["T3_pathway"])
