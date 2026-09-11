"""run_levels.py — L1-L5 certification benchmark (context.txt lines 660-684).

Trains one ensemble on the config's train split, gates it through the five
levels consecutively, and writes reports/levels/{results.json, report.md}.
A surrogate is certified at the highest level it clears WITHOUT skipping:
certified_level answers "how far can this surrogate replace explicit
simulation as the experimental substrate" with a number, not an adjective.

Usage:
    python benchmarks/run_levels.py [--full]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._common import ensure_dir, log_manifest, md_table, save_json, timed


def run(full: bool = False) -> dict:
    from virtual_kernel import VKConfig, fast_config
    from virtual_kernel.datasets import dataset_hash, train_holdout_split
    from virtual_kernel.levels import certify

    cfg = VKConfig(name="levels-full") if full else fast_config("levels-smoke")
    if full:
        cfg.n_members, cfg.epochs = 5, 300
    else:
        # Smoke must still feed the gates enough signal to be meaningful:
        # training is sub-second, so use a mid-scale config, not a starved one.
        cfg.train_seeds = list(range(7000, 7010))
        cfg.test_seeds = list(range(9000, 9004))
        cfg.steps_per_episode = 20
        cfg.n_members, cfg.epochs = 3, 120

    out, elapsed = timed(certify, cfg, verbose=False)
    out["elapsed_s"] = round(elapsed, 2)
    out["mode"] = "full" if full else "smoke"

    rep_dir = ensure_dir(os.path.join(cfg.report_dir, "levels"))
    save_json(os.path.join(rep_dir, "results.json"),
              {"config": cfg.to_dict(), **out})
    rows = [[r["level"], r["name"], r["passed"],
             {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in r["metrics"].items() if not isinstance(v, (dict, list))}]
            for r in out["results"]]
    md = ("# Levels certification — surrogate hierarchy L1-L5\n\n"
          f"config `{cfg.hash}`, mode {out['mode']}, {elapsed:.1f}s\n\n"
          f"**certified: {out['certified_name']} (L{out['certified_level']})**\n\n"
          + md_table(["level", "name", "passed", "key metrics"], rows)
          + f"\n\nfidelity: {out['fidelity']['n_passed']}/"
            f"{out['fidelity']['n_decided']} specs passed\n")
    with open(os.path.join(rep_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        train_ds, _, _, _ = train_holdout_split(
            cfg.train_seeds, cfg.test_seeds, cfg.steps_per_episode,
            cfg.arms, cfg.n_cpus, cfg.dt_s)
        from virtual_kernel import EnsembleDynamics
        log_manifest("levels", cfg, dataset_hash(train_ds), [],
                     {"certified": out["certified_name"],
                      "fidelity": out["fidelity"]["n_passed"]})
    except Exception as e:
        out["manifest_warning"] = str(e)
    return out


if __name__ == "__main__":
    full = "--full" in sys.argv
    out = run(full=full)
    print(f"certified: {out['certified_name']}")
    for r in out["results"]:
        print(f"  L{r['level']} {r['name']}: {'PASS' if r['passed'] else 'FAIL'} "
              f"{ {k: v for k, v in r['metrics'].items() if not isinstance(v, (dict, list))} }")
