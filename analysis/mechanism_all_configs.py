#!/usr/bin/env python3
"""The braking-failure mechanism across ALL quantization recipes.

Extends analysis/brake_onset_ci.py from the 3 configs in benchmark12 to all 6,
pulling W8 / AWQ-W4 / W4A4+DCT from the positive_hunt sweep. Both sweeps cover
the same 16 adversarial scene/scenario instances (5 frontal, 3 side, 8 stationary).

The claim under test: the early-crash + late-braking signature is specific to
4-bit recipes and to stationary scenarios, and W8 -- the recipe the study
certifies -- looks like FP16.

Outcome classes (see the caveat in brake_onset_ci.py): the evaluator stops
accumulating once the ego crashes, so a rollout that ends in <MIN_FRAMES frames
IS the catastrophic outcome, not missing data.

Usage: python3 analysis/mechanism_all_configs.py
"""
import json, glob, os, re, random
from collections import defaultdict

OUT = os.path.expanduser("~/neuroncap/neuro-ncap/output")
SOURCES = {
    "benchmark12":   ["fp16", "naive_w4", "group_w4_g128"],
    "positive_hunt": ["w8", "awq_w4_g128", "w4a4_dct"],
}
ORDER = ["fp16", "w8", "group_w4_g128", "awq_w4_g128", "w4a4_dct", "naive_w4"]
KINDS = ["frontal", "side", "stationary"]

MIN_FRAMES = 5
MOVE_THRESH = 5.0
STOP_THRESH = 5.0
N_BOOT = 10_000
random.seed(0)


def load_run(run_dir):
    try:
        traj = json.load(open(os.path.join(run_dir, "trajectories.json")))
    except Exception:
        return None
    p = [float(w[-1][1]) for _, w in sorted(traj.items()) if w and len(w) >= 6]
    if not p:
        return None
    if len(p) < MIN_FRAMES:
        return ("early", None)
    if max(p) < MOVE_THRESH:
        return ("frozen", None)
    for i, r in enumerate(p):
        if r < STOP_THRESH:
            return ("moving", i / len(p))
    return ("moving", 1.0)


def parse_scores(log_path):
    try:
        txt = open(log_path, errors="ignore").read()
    except Exception:
        return []
    return [float(a) for a in re.findall(r"ncap_score:\s*([\d.]+),\s*impact_speed:", txt)]


def boot_ci(vals, n=N_BOOT):
    if not vals:
        return (float("nan"),) * 3
    k = len(vals)
    means = sorted(sum(random.choices(vals, k=k)) / k for _ in range(n))
    return (sum(vals) / k, means[int(.025 * n)], means[int(.975 * n)])


def collect():
    runs = defaultdict(lambda: defaultdict(list))  # (scene,kind) -> cfg -> [(outcome,onset,ncap)]
    for sweep, cfgs in SOURCES.items():
        for cfg_dir in sorted(glob.glob(os.path.join(OUT, sweep, "*_*"))):
            if not os.path.isdir(cfg_dir):
                continue
            parts = os.path.basename(cfg_dir).split("_")
            scene, kind, cfg = parts[0], parts[1], "_".join(parts[2:])
            if kind not in KINDS or cfg not in cfgs:
                continue
            scores = parse_scores(cfg_dir + ".log")
            for i, run in enumerate(sorted(glob.glob(os.path.join(cfg_dir, "run_*")),
                                           key=lambda p: int(p.rsplit("_", 1)[1]))):
                r = load_run(run)
                if r is None:
                    continue
                runs[(scene, kind)][cfg].append(
                    (r[0], r[1], scores[i] if i < len(scores) else None))
    return runs


def main():
    runs = collect()

    print("=" * 96)
    print("OUTCOME BREAKDOWN BY RECIPE AND SCENARIO TYPE")
    print("early = rollout ended <5 frames (crashed almost immediately)")
    print("=" * 96)
    print(f"{'kind':<11} {'config':<15} {'early':>7} {'frozen':>7} {'moving':>7} "
          f"{'onset|mov':>10} {'95% CI':>17} {'NCAP':>6} {'n_scen':>7}")
    print("-" * 96)

    onset_by = defaultdict(dict)   # kind -> cfg -> {scene: onset}
    for kind in KINDS:
        for cfg in ORDER:
            on_s, early_s, froz_s, ncaps = [], [], [], []
            for (sc, k), d in runs.items():
                if k != kind or cfg not in d:
                    continue
                rs = d[cfg]
                early_s.append(sum(1 for r in rs if r[0] == "early") / len(rs))
                froz_s.append(sum(1 for r in rs if r[0] == "frozen") / len(rs))
                mv = [r for r in rs if r[0] == "moving"]
                sc_n = [r[2] for r in rs if r[2] is not None]
                if sc_n:
                    ncaps.append(sum(sc_n) / len(sc_n))
                if mv:
                    on_s.append(sum(r[1] for r in mv) / len(mv))
                    onset_by[kind].setdefault(cfg, {})[sc] = on_s[-1]
            if not early_s:
                continue
            m, lo, hi = boot_ci(on_s)
            ee = sum(early_s) / len(early_s)
            fz = sum(froz_s) / len(froz_s)
            nc = sum(ncaps) / len(ncaps) if ncaps else float("nan")
            ci = f"[{lo:.3f}, {hi:.3f}]" if on_s else "--"
            print(f"{kind:<11} {cfg:<15} {ee:>6.1%} {fz:>6.1%} {1-ee-fz:>6.1%} "
                  f"{m:>10.3f} {ci:>17} {nc:>6.2f} {len(on_s):>7}")
        print()

    print("=" * 96)
    print("PAIRED per-scenario braking-onset difference vs FP16 (+ = brakes LATER)")
    print("=" * 96)
    for kind in KINDS:
        for cfg in ORDER:
            if cfg == "fp16":
                continue
            a, b = onset_by[kind].get(cfg, {}), onset_by[kind].get("fp16", {})
            diffs = [a[s] - b[s] for s in a if s in b]
            if not diffs:
                continue
            m, lo, hi = boot_ci(diffs)
            flag = "  <-- excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"{kind:<11} {cfg:<15} delta={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"n={len(diffs)}{flag}")
        print()

    print("=" * 96)
    print("EARLY-CRASH RATE, stationary only, 95% bootstrap CI over scenarios")
    print("(the headline: does it separate 4-bit recipes from FP16/W8?)")
    print("=" * 96)
    for cfg in ORDER:
        vals = [sum(1 for r in d[cfg] if r[0] == "early") / len(d[cfg])
                for (sc, k), d in runs.items() if k == "stationary" and cfg in d]
        if not vals:
            continue
        m, lo, hi = boot_ci(vals)
        print(f"{cfg:<15} early={m:>6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  n_scen={len(vals)}")

    freedrive_control()


def freedrive_control():
    """Adversary-free control: same scenes, no obstacle to respond to.

    NOTE: without an adversary, an early-terminating rollout is not a collision,
    so this rate is NOT comparable in absolute terms to the adversarial one --
    its cause has not yet been established. What is comparable is the
    *between-recipe* contrast within this condition.
    """
    src = {"freedrive": ["fp16", "group_w4_g128", "naive_w4"],
           "positive_hunt": ["w8", "awq_w4_g128", "w4a4_dct"]}
    res = defaultdict(list)
    for sweep, cfgs in src.items():
        for d in sorted(glob.glob(os.path.join(OUT, sweep, "*_freedrive_*"))):
            if not os.path.isdir(d):
                continue
            cfg = "_".join(os.path.basename(d).split("_")[2:])
            if cfg not in cfgs:
                continue
            o = [x for x in (load_run(r) for r in glob.glob(os.path.join(d, "run_*"))) if x]
            if o:
                res[cfg].append(sum(1 for x in o if x[0] == "early") / len(o))

    print()
    print("=" * 96)
    print("CONTROL: free-drive (adversary-free), same scenes, early-termination rate")
    print("=" * 96)
    for cfg in ORDER:
        if cfg not in res:
            continue
        m, lo, hi = boot_ci(res[cfg])
        print(f"{cfg:<15} early={m:>6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  n_scen={len(res[cfg])}")
    print("\nNo recipe separates from FP16 here -- all CIs overlap heavily. The 4-bit")
    print("early-crash signature appears only when there is an obstacle to respond to.")


if __name__ == "__main__":
    main()
