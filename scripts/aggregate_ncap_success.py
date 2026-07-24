#!/usr/bin/env python3
"""Aggregate NeuroNCAP closed-loop *success* (impact-speed-graded NCAP score, 0-5)
and collision rate across a benchmark run, with bootstrap confidence intervals.

This replaces `generation coherence` (a well-formedness proxy) with the real
closed-loop safety metric NeuroNCAP was designed around: the 5-star score, which
is graded by impact speed and therefore discriminates configs even when the
binary collision rate saturates.

Usage:
    python scripts/aggregate_ncap_success.py \
        --root /home/justin_williams1/neuroncap/neuro-ncap/output/benchmark12

Directory layout expected (per NeuroNCAP):
    <root>/<scene>_<scenario>_<config>/run_<seed>/metrics.json   (or .../metrics.json)
where <config> in {fp16, naive_w4, group_w4_g128, ...}. Any config suffix is
auto-detected; pass --configs to fix an order.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

# scene_scenario  +  config, where config is the trailing token(s) we know about.
KNOWN_CONFIGS = ["fp16", "naive_w4", "group_w4_g128", "group_w4_g64",
                 "w4a4", "w4a4_dct", "w16a4", "w16a4_dct"]


def parse_dir(name: str, configs: list[str]) -> tuple[str, str] | None:
    for cfg in sorted(configs, key=len, reverse=True):
        suf = "_" + cfg
        if name.endswith(suf):
            return name[: -len(suf)], cfg
    return None


def load_scores(root: str, configs: list[str]):
    """scenario -> config -> {'ncap': [...], 'coll': [...]}"""
    data: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"ncap": [], "coll": []})
    )
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        parsed = parse_dir(d, configs)
        if parsed is None:
            continue
        scen, cfg = parsed
        mjs = glob.glob(os.path.join(full, "run_*", "metrics.json"))
        if not mjs:
            mjs = glob.glob(os.path.join(full, "metrics.json"))
        for mj in mjs:
            try:
                j = json.load(open(mj))
            except Exception:
                continue
            if "ncap_score" not in j:
                continue
            data[scen][cfg]["ncap"].append(float(j["ncap_score"]))
            c = j.get("any_collide@0.0s")
            if c is not None:
                data[scen][cfg]["coll"].append(1.0 if c else 0.0)
    return data


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(values, dtype=float)
    rng = rng or np.random.default_rng(0)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean()), float(lo), float(hi)


def scenario_level_means(data, cfg, key):
    """One number per scenario (its seed-mean), so scenarios are weighted equally
    -- matches how the coherence table was aggregated."""
    out = []
    for scen in sorted(data):
        vals = data[scen][cfg][key]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--configs", nargs="*", default=KNOWN_CONFIGS)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data = load_scores(args.root, args.configs)
    present = [c for c in args.configs
               if any(data[s][c]["ncap"] for s in data)]

    print(f"# NeuroNCAP closed-loop success  (root={args.root})")
    print(f"# NCAP score 0-5 (impact-speed graded, higher=safer); "
          f"scenario-level paired means, 95% bootstrap CI over scenarios\n")

    print(f"{'config':16s} {'nrollouts':>9s} {'meanNCAP':>9s} "
          f"{'95% CI':>16s} {'collrate':>9s}")
    for cfg in present:
        scen_ncap = scenario_level_means(data, cfg, "ncap")
        n_roll = sum(len(data[s][cfg]["ncap"]) for s in data)
        m, lo, hi = bootstrap_ci(scen_ncap, args.n_boot, rng=rng)
        scen_coll = scenario_level_means(data, cfg, "coll")
        cm, _, _ = bootstrap_ci(scen_coll, args.n_boot, rng=rng)
        print(f"{cfg:16s} {n_roll:9d} {m:9.3f} "
              f"  [{lo:5.2f}, {hi:5.2f}] {cm:8.1%}")

    print(f"\n# Per-scenario mean NCAP score")
    hdr = "scenario".ljust(20) + "".join(f"{c:>16s}" for c in present)
    print(hdr)
    for scen in sorted(data):
        row = scen.ljust(20)
        for cfg in present:
            vals = data[scen][cfg]["ncap"]
            row += (f"{np.mean(vals):16.2f}" if vals else f"{'-':>16s}")
        print(row)

    # Paired delta: group vs naive, per scenario (the thesis test)
    if "naive_w4" in present and "group_w4_g128" in present:
        print(f"\n# Thesis test: group_w4_g128 - naive_w4 per scenario "
              f"(paired), + bootstrap CI on the mean paired difference")
        diffs = []
        for scen in sorted(data):
            n = data[scen]["naive_w4"]["ncap"]
            g = data[scen]["group_w4_g128"]["ncap"]
            if n and g:
                d = float(np.mean(g) - np.mean(n))
                diffs.append(d)
                print(f"  {scen:22s} {d:+6.2f}")
        m, lo, hi = bootstrap_ci(diffs, args.n_boot, rng=rng)
        print(f"  {'MEAN paired diff':22s} {m:+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
        verdict = ("group RECOVERS (CI excludes 0, positive)" if lo > 0 else
                   "group WORSE (CI excludes 0, negative)" if hi < 0 else
                   "NO DIFFERENCE (CI spans 0) -- granularity does not restore NCAP success")
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
