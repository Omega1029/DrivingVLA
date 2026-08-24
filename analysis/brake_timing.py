#!/usr/bin/env python3
"""Time-resolved: WHEN does the planner commit to braking, and does it predict outcome?

Round 1 showed group-W4's aggregate planned reach in stationary scenarios is
identical to FP16 (17.77 vs 17.77) yet its NCAP score is 2.62 vs 4.43. So the
failure is not in gross trajectory statistics. Test whether it is in *timing*:
braking onset frame, and whether onset predicts the per-run NCAP score.
"""
import json, glob, os, re
from collections import defaultdict

ROOT = os.path.expanduser("~/neuroncap/neuro-ncap/output/benchmark12")
CONFIGS = ["fp16", "naive_w4", "group_w4_g128"]


def reach_profile(run_dir):
    try:
        traj = json.load(open(os.path.join(run_dir, "trajectories.json")))
    except Exception:
        return None
    out = []
    for _, wps in sorted(traj.items()):
        if wps and len(wps) >= 6:
            out.append(float(wps[-1][1]))
    return out or None


def parse_scores(log_path):
    """Per-run ncap_score / impact_speed, in run order."""
    try:
        txt = open(log_path, errors="ignore").read()
    except Exception:
        return []
    return [(float(a), float(b)) for a, b in
            re.findall(r"ncap_score:\s*([\d.]+),\s*impact_speed:\s*([\d.]+)", txt)]


def onset(profile, thresh=5.0):
    """Fraction of the way through the rollout at which planned reach first
    drops below thresh (i.e. commits to stopping). 1.0 = never."""
    for i, r in enumerate(profile):
        if r < thresh:
            return i / len(profile)
    return 1.0


def main():
    rows = defaultdict(lambda: defaultdict(list))
    for cfg_dir in sorted(glob.glob(os.path.join(ROOT, "*_*"))):
        if not os.path.isdir(cfg_dir):
            continue
        base = os.path.basename(cfg_dir)
        parts = base.split("_")
        scene, kind, cfg = parts[0], parts[1], "_".join(parts[2:])
        if cfg not in CONFIGS:
            continue
        scores = parse_scores(cfg_dir + ".log")
        runs = sorted(glob.glob(os.path.join(cfg_dir, "run_*")),
                      key=lambda p: int(p.rsplit("_", 1)[1]))
        for i, run in enumerate(runs):
            p = reach_profile(run)
            if not p:
                continue
            sc = scores[i][0] if i < len(scores) else None
            rows[kind][cfg].append((onset(p), sc))

    print(f"{'kind':<12} {'config':<16} {'brake_onset':>12} {'never_brakes':>13} {'mean_ncap':>10}")
    print("-" * 68)
    for kind in sorted(rows):
        for cfg in CONFIGS:
            rs = [r for r in rows[kind].get(cfg, [])]
            if not rs:
                continue
            n = len(rs)
            onsets = [r[0] for r in rs]
            never = sum(1 for o in onsets if o >= 1.0) / n
            scored = [r[1] for r in rs if r[1] is not None]
            ms = sum(scored) / len(scored) if scored else float("nan")
            print(f"{kind:<12} {cfg:<16} {sum(onsets)/n:>12.3f} {never:>12.1%} {ms:>10.2f}")
        print()

    # Within stationary+group_w4: does late braking predict a bad score?
    print("=" * 68)
    print("Within-config: brake onset vs NCAP score (stationary scenarios)")
    print("=" * 68)
    for cfg in CONFIGS:
        rs = [r for r in rows["stationary"].get(cfg, []) if r[1] is not None]
        if len(rs) < 4:
            continue
        early = [r[1] for r in rs if r[0] < 0.5]
        late = [r[1] for r in rs if r[0] >= 0.5]
        f = lambda v: f"{sum(v)/len(v):.2f} (n={len(v)})" if v else "n/a"
        print(f"{cfg:<16} early-brake NCAP {f(early):<18} late/never NCAP {f(late)}")


if __name__ == "__main__":
    main()
