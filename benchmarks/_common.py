"""Shared benchmark harness: JSON + Markdown reports, manifest logging."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: dict) -> str:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    return path


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def log_manifest(name: str, config, dataset_hash: str, model_hashes,
                 results: dict, manifest_path: str = "reports/manifests.jsonl"):
    from virtual_kernel import ManifestLog, RunManifest
    log = ManifestLog(manifest_path)
    m = RunManifest(name=name, config_hash=config.hash,
                    dataset_hash=dataset_hash, model_hashes=list(model_hashes),
                    results=results)
    return log.append(m)


def timed(fn, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    return out, time.time() - t0
