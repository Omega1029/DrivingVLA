#!/usr/bin/env python3
"""Braking-onset analysis with bootstrap CIs, matching the paper's protocol.

Improvements over the first pass (analysis/brake_timing.py):
  * Freeze detection. A run whose planned reach never exceeds MOVE_THRESH never
    intended to move, so its "early braking" is an artifact of the malformed-text
    collapse, not a braking decision. Those runs are excluded from onset stats and
    reported separately as a freeze rate.
  * Percentile-bootstrap CIs over scenario-level means (10^4 resamples), and
    paired per-scenario differences vs FP16 -- the same treatment the graded-safety
    table uses.
  * Spearman correlation between onset and per-run NCAP score.

Usage: python3 analysis/brake_onset_ci.py
"""
import json, glob, os, re, random
from collections import defaultdict

ROOT = os.path.expanduser("~/neuroncap/neuro-ncap/output/benchmark12")
CONFIGS = ["fp16", "naive_w4", "group_w4_g128"]
MOVE_THRESH = 5.0   # a run must plan to travel this far (m) at some point to count as "moving"
STOP_THRESH = 5.0   # planned 3s reach below this = committed to stopping
N_BOOT = 10_000
random.seed(0)      # deterministic


MIN_FRAMES = 5  # below this the rollout ended early -- the evaluator stops
                # accumulating once the ego has crashed, so a 1-4 frame rollout
                # IS the catastrophic outcome, not missing data. Filtering these
                # out would be survivorship bias against exactly the worst runs.


def reach_profile(run_dir):
    """Returns (profile, early_termination). profile may be short."""
    try:
        traj = json.load(open(os.path.join(run_dir, "trajectories.json")))
    except Exception:
        return None, False
    p = [float(w[-1][1]) for _, w in sorted(traj.items()) if w and len(w) >= 6]
    if not p:
        return None, False
    return p, len(p) < MIN_FRAMES


def parse_scores(log_path):
    try:
        txt = open(log_path, errors="ignore").read()
    except Exception:
        return []
    return [float(a) for a in
            re.findall(r"ncap_score:\s*([\d.]+),\s*impact_speed:", txt)]


def classify(profile, early):
    """Returns (outcome, onset).

    outcome is one of:
      'early'  -- rollout ended within MIN_FRAMES: the ego crashed almost immediately
      'frozen' -- survived but never planned to move at all
      'moving' -- planned to move; onset says when it committed to stopping
    """
    if early:
        return "early", None
    if max(profile) < MOVE_THRESH:
        return "frozen", None
    for i, r in enumerate(profile):
        if r < STOP_THRESH:
            return "moving", i / len(profile)
    return "moving", 1.0


def boot_ci(vals, n=N_BOOT):
    if not vals:
        return (float("nan"),) * 3
    k = len(vals)
    means = sorted(sum(random.choices(vals, k=k)) / k for _ in range(n))
    return (sum(vals) / k, means[int(.025 * n)], means[int(.975 * n)])


def spearman(xs, ys):
    if len(xs) < 4:
        return float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else float("nan")


def main():
    # (scene, kind) -> cfg -> list of (frozen, onset, ncap)
    runs = defaultdict(lambda: defaultdict(list))
    for cfg_dir in sorted(glob.glob(os.path.join(ROOT, "*_*"))):
        if not os.path.isdir(cfg_dir):
            continue
        parts = os.path.basename(cfg_dir).split("_")
        scene, kind, cfg = parts[0], parts[1], "_".join(parts[2:])
        if cfg not in CONFIGS:
            continue
        scores = parse_scores(cfg_dir + ".log")
        for i, run in enumerate(sorted(glob.glob(os.path.join(cfg_dir, "run_*")),
                                       key=lambda p: int(p.rsplit("_", 1)[1]))):
            p, early = reach_profile(run)
            if p is None:
                continue
            outcome, onset = classify(p, early)
            runs[(scene, kind)][cfg].append(
                (outcome, onset, scores[i] if i < len(scores) else None))

    # ---- scenario-level means, then bootstrap over scenarios ----
    print("OUTCOME BREAKDOWN + braking onset, 95% percentile bootstrap over scenarios")
    print("early = rollout ended <5 frames (crashed almost immediately)")
    print(f"{'kind':<12} {'config':<16} {'early':>7} {'frozen':>7} {'moving':>7} "
          f"{'onset|moving':>13} {'95% CI':>17} {'n_scen':>7}")
    print("-" * 92)
    per_kind = defaultdict(lambda: defaultdict(dict))
    for kind in ["frontal", "side", "stationary"]:
        for cfg in CONFIGS:
            on_s, early_s, froz_s = [], [], []
            for (sc, k), d in runs.items():
                if k != kind or cfg not in d:
                    continue
                rs = d[cfg]
                mv = [r for r in rs if r[0] == "moving"]
                early_s.append(sum(1 for r in rs if r[0] == "early") / len(rs))
                froz_s.append(sum(1 for r in rs if r[0] == "frozen") / len(rs))
                if mv:
                    on_s.append(sum(r[1] for r in mv) / len(mv))
                per_kind[kind][cfg][sc] = (on_s[-1] if mv else None)
            m, lo, hi = boot_ci(on_s)
            ee, _, _ = boot_ci(early_s)
            fz, _, _ = boot_ci(froz_s)
            print(f"{kind:<12} {cfg:<16} {ee:>6.1%} {fz:>6.1%} {1-ee-fz:>6.1%} "
                  f"{m:>13.3f} {'['+format(lo,'.3f')+', '+format(hi,'.3f')+']':>17} {len(on_s):>7}")
        print()

    # ---- paired per-scenario difference vs FP16 ----
    print("=" * 82)
    print("PAIRED per-scenario onset difference vs FP16 (positive = brakes LATER)")
    print("=" * 82)
    for kind in ["frontal", "side", "stationary"]:
        for cfg in CONFIGS[1:]:
            diffs = [per_kind[kind][cfg][sc] - per_kind[kind]["fp16"][sc]
                     for sc in per_kind[kind][cfg]
                     if per_kind[kind][cfg].get(sc) is not None
                     and per_kind[kind]["fp16"].get(sc) is not None]
            if not diffs:
                continue
            m, lo, hi = boot_ci(diffs)
            sig = "  <-- excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"{kind:<12} {cfg:<16} delta={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(diffs)}{sig}")
        print()

    # ---- onset vs NCAP, within config ----
    print("=" * 82)
    print("Spearman rho: braking onset vs per-run NCAP score (non-frozen runs)")
    print("=" * 82)
    for kind in ["frontal", "side", "stationary"]:
        for cfg in CONFIGS:
            pts = [(r[1], r[2]) for (sc, k), d in runs.items() if k == kind
                   for r in d.get(cfg, []) if r[0] == "moving" and r[2] is not None]
            if len(pts) < 8:
                continue
            print(f"{kind:<12} {cfg:<16} rho={spearman([p[0] for p in pts], [p[1] for p in pts]):+.3f}  n={len(pts)}")
        print()


if __name__ == "__main__":
    main()
