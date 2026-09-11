"""run_degradation.py — publishable robustness studies (context.txt §'Those
are publishable questions'):

  Q1  How much of the exact dynamics can a surrogate recover? (data scale)
  Q2  How far can it roll out before divergence? (horizon sweep)
  Q3  Does search compensate for model error? (ranking fidelity vs MSE as
      training data shrinks — Spearman should decay slower than MSE grows)
  Q4  When does uncertainty become dangerous? (calibration vs data scale)

Degradations: training-episode count + target noise sigma + action-coverage
thinning (drop arms). Each cell trains a small ensemble and scores it.

Usage:  python benchmarks/run_degradation.py [--full]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._common import ensure_dir, log_manifest, md_table, save_json, timed


def run(full: bool = False) -> dict:
    import numpy as np
    from kernel.policy.schemas import PolicyAction, SchedulerAction
    from kernel.simulator.env import KernelSimulator
    from virtual_kernel import (
        VKConfig, EnsembleDynamics, TruthOracle, fast_config,
        divergence_stats, spearman_rank, calibration as cal_fn,
        virtual_screen,
    )
    from virtual_kernel.datasets import capture_traces, to_transitions

    cfg = VKConfig(name="degradation-full" if full else "degradation-smoke") if full \
        else fast_config("degradation-smoke")
    episode_grid = [16, 8, 4] if full else [4, 2]
    noise_grid = [0.0, 0.05] if full else [0.0]
    horizons = [1, 2, 4, 8] if full else [1, 2, 4]
    epochs = 120 if full else cfg.epochs

    kir0 = KernelSimulator(seed=11, n_cpus=cfg.n_cpus).current_kir()
    cands = [[a, b] for a in (2000, 6000, 12000) for b in (2000, 6000, 12000)]

    cells = []
    for n_ep in episode_grid:
        for sigma in noise_grid:
            eps = capture_traces(n_ep, cfg.steps_per_episode, base_seed=cfg.seed,
                                 arms=cfg.arms, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)
            ds = to_transitions(eps, cfg.dt_s)
            if sigma > 0:
                rng = np.random.default_rng(0)
                ds.targets = ds.targets + rng.normal(0, sigma, size=ds.targets.shape)
                ds.targets = np.clip(ds.targets, 0, 1)
            ens = EnsembleDynamics.train(n_members=2, epochs=epochs,
                                         n_cpus=cfg.n_cpus, dataset=ds)
            ev = ens.evaluate(ds)
            cal = ens.calibration(ds)

            # horizon divergence on a fixed action pattern
            oracle = TruthOracle(seed=11, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)
            acts = [PolicyAction(policy_id="d", scheduler=SchedulerAction(
                target_latency_us=6000))] * max(horizons)
            rep = oracle.evaluate(ens, kir0, acts,
                                  divergence_threshold=cfg.divergence_threshold)
            curves = {h: rep.rollout_mae_curve[:h] for h in horizons}
            divs = {h: divergence_stats(c, cfg.divergence_threshold) for h, c in curves.items()}

            # ranking fidelity over candidate 2-step sequences
            pred_r, true_r = [], []
            for seq in cands:
                pr, _ = __import__("virtual_kernel.planning", fromlist=[
                    "rollout_sequence_reward"]).rollout_sequence_reward(ens, kir0, seq)
                pred_r.append(pr)
                true_r.append(oracle.sequence_reward(kir0, seq))
            rho = spearman_rank(pred_r, true_r)

            cells.append({
                "n_episodes": n_ep, "noise_sigma": sigma,
                "single_step_mse": round(ev["mse"], 5),
                "mean_disagreement": round(ev["mean_disagreement"], 5),
                "divergence_step_h8" if full else "divergence_step": (
                    divs[max(horizons)]["divergence_step"]),
                "final_drift": round(divs[max(horizons)]["final"], 5),
                "spearman_ranking": round(rho, 4) if rho == rho else None,
                "calibration_corr": (round(cal["uncertainty_error_corr"], 4)
                                     if cal["uncertainty_error_corr"] == cal["uncertainty_error_corr"] else None),
            })
    out = {"config": cfg.to_dict(), "config_hash": cfg.hash, "cells": cells}
    rep_dir = ensure_dir(os.path.join(cfg.report_dir, "degradation"))
    save_json(os.path.join(rep_dir, "results.json"), out)
    md = ("# Degradation studies\n\n"
          "Q1 data scale / Q2 horizon divergence / Q3 ranking-vs-MSE / Q4 calibration.\n\n"
          + md_table(["episodes", "noise", "mse", "disagreement", "div_step",
                      "final_drift", "spearman", "cal_corr"],
                     [[c["n_episodes"], c["noise_sigma"], c["single_step_mse"],
                       c["mean_disagreement"],
                       c.get("divergence_step_h8", c.get("divergence_step")),
                       c["final_drift"], c["spearman_ranking"],
                       c["calibration_corr"]] for c in cells]) + "\n")
    with open(os.path.join(rep_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        log_manifest("degradation", cfg, "varied", [], {"cells": cells})
    except Exception as e:
        out["manifest_warning"] = str(e)
    return out


if __name__ == "__main__":
    out = run(full="--full" in sys.argv)
    for c in out["cells"]:
        print(c)
