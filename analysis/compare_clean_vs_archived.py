#!/usr/bin/env python3
"""Compare the clean-harness re-measurement against the archived (confounded) sweep.

The archived sweeps ran configs in a fixed order against ONE renderer per scene, so only the
first config per scene saw a clean actor set. This aggregates the restore-protocol re-run and
puts it beside the archive, per scenario instance and in aggregate, with bootstrap CIs over
scenario instances (same protocol as the paper's graded-safety table).
"""
import glob, os, re, random, sys
from collections import defaultdict

CLEAN = os.path.expanduser("~/neuroncap/neuro-ncap/output/remeasure_clean")
ARCH = os.path.expanduser("~/neuroncap/neuro-ncap/output/benchmark12")
ORDER = ["fp16", "w8", "naive_w4", "group_w4_g128"]
N_BOOT = 10_000
random.seed(0)


def scores(log):
    try:
        txt = open(log, errors="ignore").read()
    except Exception:
        return []
    return [float(x) for x in re.findall(r"ncap_score:\s*([\d.]+),\s*impact_speed:", txt)]


def boot(vals):
    if not vals:
        return (float("nan"),) * 3
    k = len(vals)
    m = sorted(sum(random.choices(vals, k=k)) / k for _ in range(N_BOOT))
    return (sum(vals) / k, m[int(.025 * N_BOOT)], m[int(.975 * N_BOOT)])


def collect(root):
    out = defaultdict(dict)   # (scene,cat) -> cfg -> [scores]
    for log in glob.glob(os.path.join(root, "*_*.log")):
        b = os.path.basename(log)[:-4]
        parts = b.split("_")
        if len(parts) < 3 or parts[0] in ("rend", "model"):
            continue
        scene, cat, cfg = parts[0], parts[1], "_".join(parts[2:])
        s = scores(log)
        if s:
            out[(scene, cat)][cfg] = s
    return out


def main():
    clean, arch = collect(CLEAN), collect(ARCH)
    if not clean:
        print("no clean results yet"); return

    print(f"{'scene/scenario':<22} " + " ".join(f"{c:>15}" for c in ORDER))
    print("-" * (22 + 16 * len(ORDER)))
    for k in sorted(clean):
        row = f"{k[0]+' '+k[1]:<22} "
        for c in ORDER:
            v = clean[k].get(c)
            row += f"{(sum(v)/len(v) if v else float('nan')):>15.2f} "
        print(row)

    print("\n" + "=" * 78)
    print("AGGREGATE over scenario instances, 95% bootstrap CI")
    print("=" * 78)
    print(f"{'config':<16} {'CLEAN mean':>11} {'95% CI':>18} {'ARCHIVED':>10} {'n':>4}")
    print("-" * 78)
    agg = {}
    for c in ORDER:
        cv = [sum(v[c]) / len(v[c]) for v in clean.values() if c in v]
        av = [sum(v[c]) / len(v[c]) for v in arch.values() if c in v]
        if not cv:
            continue
        m, lo, hi = boot(cv)
        agg[c] = cv
        a = f"{sum(av)/len(av):.2f}" if av else "--"
        print(f"{c:<16} {m:>11.2f} {'['+format(lo,'.2f')+', '+format(hi,'.2f')+']':>18} {a:>10} {len(cv):>4}")

    if "fp16" in agg:
        print("\nPaired difference vs FP16 (clean harness), 95% CI over scenario instances:")
        base = {k: sum(v["fp16"]) / len(v["fp16"]) for k, v in clean.items() if "fp16" in v}
        for c in ORDER:
            if c == "fp16":
                continue
            d = [sum(v[c]) / len(v[c]) - base[k] for k, v in clean.items() if c in v and k in base]
            if not d:
                continue
            m, lo, hi = boot(d)
            verdict = "INDISTINGUISHABLE from FP16" if lo <= 0 <= hi else "DIFFERS from FP16"
            print(f"  {c:<16} delta={m:+.2f}  CI [{lo:+.2f}, {hi:+.2f}]  n={len(d)}  -> {verdict}")


if __name__ == "__main__":
    main()
