#!/usr/bin/env python3
"""Does 4-bit quantization make the driving planner under-brake?

Hypothesis for the structured failure (all 4-bit recipes fail stationary-obstacle
scenarios, pass frontal): quantized backbones fail to commit to a stopping plan.

Per frame the planner emits 6 waypoints [lateral, longitudinal] over a 3s horizon.
The final waypoint's longitudinal value is the planned 3s reach = proxy for
planned speed. Braking = driving that reach toward 0.
"""
import json, sys, glob, os
from collections import defaultdict

ROOT = os.path.expanduser("~/neuroncap/neuro-ncap/output/benchmark12")


def run_stats(d):
    """Return (mean_reach, min_reach, frac_frames_committed_stop) for one rollout."""
    try:
        traj = json.load(open(os.path.join(d, "trajectories.json")))
    except Exception:
        return None
    reaches = []
    for _, wps in sorted(traj.items()):
        if not wps or len(wps) < 6:
            continue
        reaches.append(float(wps[-1][1]))  # longitudinal reach at 3s
    if len(reaches) < 5:
        return None
    committed = sum(1 for r in reaches if r < 2.0) / len(reaches)
    return (sum(reaches) / len(reaches), min(reaches), committed)


def main():
    # scenario -> config -> list of per-run stats
    data = defaultdict(lambda: defaultdict(list))
    for cfg_dir in sorted(glob.glob(os.path.join(ROOT, "*_*"))):
        if not os.path.isdir(cfg_dir):
            continue
        base = os.path.basename(cfg_dir)
        # e.g. 0099_stationary_group_w4_g128
        parts = base.split("_")
        scene, kind = parts[0], parts[1]
        cfg = "_".join(parts[2:])
        for run in sorted(glob.glob(os.path.join(cfg_dir, "run_*"))):
            s = run_stats(run)
            if s:
                data[(scene, kind)][cfg].append(s)

    configs = ["fp16", "naive_w4", "group_w4_g128"]
    by_kind = defaultdict(lambda: defaultdict(list))

    print(f"{'scenario':<22} {'config':<16} {'mean_reach':>10} {'min_reach':>10} {'stop_frac':>10}")
    print("-" * 72)
    for (scene, kind) in sorted(data):
        for cfg in configs:
            runs = data[(scene, kind)].get(cfg, [])
            if not runs:
                continue
            n = len(runs)
            mr = sum(r[0] for r in runs) / n
            mn = sum(r[1] for r in runs) / n
            sf = sum(r[2] for r in runs) / n
            by_kind[kind][cfg].append((mr, mn, sf))
            print(f"{scene+'_'+kind:<22} {cfg:<16} {mr:>10.2f} {mn:>10.2f} {sf:>10.2%}")

    print("\n" + "=" * 72)
    print("AGGREGATE BY SCENARIO TYPE (mean over scenarios)")
    print("=" * 72)
    print(f"{'kind':<14} {'config':<16} {'mean_reach':>10} {'min_reach':>10} {'stop_frac':>10}")
    print("-" * 72)
    for kind in sorted(by_kind):
        for cfg in configs:
            rows = by_kind[kind].get(cfg, [])
            if not rows:
                continue
            n = len(rows)
            print(f"{kind:<14} {cfg:<16} "
                  f"{sum(r[0] for r in rows)/n:>10.2f} "
                  f"{sum(r[1] for r in rows)/n:>10.2f} "
                  f"{sum(r[2] for r in rows)/n:>10.2%}")
        print()


if __name__ == "__main__":
    main()
