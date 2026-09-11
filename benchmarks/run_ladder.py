"""run_ladder.py — the killer benchmark (context.txt Virtual Chess §6).

Compares four systems under a FIXED oracle-interaction budget B:
  rung 1  heuristic policy on exact dynamics
  rung 2  exact exhaustive search, capped at B oracle calls
  rung 3  surrogate beam search (0 virtual-cost) + 1 oracle validation
  rung 4  surrogate MCTS (0 virtual-cost) + 1 oracle validation

Metric: strength achieved per oracle interaction. The surrogate rungs must
win on reward-per-call by searching thousands of virtual worlds per probe.

Usage:
    python benchmarks/run_ladder.py [--full]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._common import ensure_dir, log_manifest, md_table, save_json, timed


def run(full: bool = False) -> dict:
    from virtual_kernel import (
        VKConfig, EnsembleDynamics, TruthOracle, fast_config, run_ladder,
    )
    from virtual_kernel.datasets import capture_traces, to_transitions
    from kernel.simulator.env import KernelSimulator

    cfg = VKConfig(name="ladder-full" if full else "ladder-smoke") if full \
        else fast_config("ladder-smoke")
    horizon = 3 if full else 2
    mcts_sims = 300 if full else 40

    ens, t_train = timed(
        EnsembleDynamics.train, n_members=cfg.n_members,
        n_episodes=cfg.n_episodes, steps_per_episode=cfg.steps_per_episode,
        base_seed=cfg.seed, epochs=cfg.epochs, n_cpus=cfg.n_cpus)
    kir0 = KernelSimulator(seed=11, n_cpus=cfg.n_cpus).current_kir()

    report, t_ladder = timed(
        run_ladder, ens, kir0, horizon=horizon, budget=cfg.oracle_budget,
        arms=cfg.arms, mcts_sims=mcts_sims, beam_width=cfg.beam_width,
        seed=cfg.seed, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)

    best = max(r.oracle_reward for r in report.rungs)
    rows = []
    for r in report.rungs:
        rpc = r.oracle_reward / max(r.oracle_calls, 1)
        rows.append({"rung": r.name, "latencies": r.latencies,
                     "oracle_reward": round(r.oracle_reward, 4),
                     "oracle_calls": r.oracle_calls,
                     "reward_per_call": round(rpc, 4),
                     "regret_frac": round((best - r.oracle_reward) / max(abs(best), 1e-9), 4),
                     "predicted_reward": (round(r.predicted_reward, 4)
                                          if r.predicted_reward is not None else None)})
    out = {"config": cfg.to_dict(), "config_hash": cfg.hash,
           "horizon": horizon, "train_s": round(t_train, 2),
           "ladder_s": round(t_ladder, 2), "rungs": rows,
           "winner": report.winner().name}
    rep_dir = ensure_dir(os.path.join(cfg.report_dir, "ladder"))
    save_json(os.path.join(rep_dir, "results.json"), out)
    md = ("# Ladder benchmark — strength per oracle interaction\n\n"
          f"config `{cfg.hash}`, horizon {horizon}\n\n"
          + md_table(["rung", "latencies", "oracle_reward", "calls",
                      "reward/call", "regret_frac", "predicted"],
                     [[r["rung"], r["latencies"], r["oracle_reward"],
                       r["oracle_calls"], r["reward_per_call"],
                       r["regret_frac"], r["predicted_reward"]] for r in rows])
          + f"\n\nwinner: **{out['winner']}**\n")
    with open(os.path.join(rep_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        from virtual_kernel.datasets import dataset_hash
        log_manifest("ladder", cfg,
                     dataset_hash(ens.dataset) if ens.dataset is not None else "none",
                     ens.hashes(), {"winner": out["winner"], "rungs": rows})
    except Exception as e:
        out["manifest_warning"] = str(e)
    return out


if __name__ == "__main__":
    full = "--full" in sys.argv
    out = run(full=full)
    print(f"winner: {out['winner']}")
    for r in out["rungs"]:
        print(f"  {r['rung']}: reward={r['oracle_reward']} calls={r['oracle_calls']} "
              f"rpc={r['reward_per_call']} seq={r['latencies']}")
