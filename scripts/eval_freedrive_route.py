#!/usr/bin/env python3
"""Offline free-drive route metric for NeuroNCAP closed-loop runs.

Computes, per rollout, from what the harness already dumps (ego_poses.json,
reference_trajectory.json, metrics.json):

  * route-following L2   : driven world (x,y) vs the logged reference route,
                           interpolated to each driven timestamp.
  * route SUCCESS (defn b): reached the route end (no early collision termination)
                           AND stayed within the corridor (max drift <= tau_fail)
                           AND no collision. Completion-style, carries over to
                           route datasets (nuPlan/CARLA-style route success).
  * off-corridor failure : fraction of rollouts whose drift ever exceeds tau_fail.

Aggregates by config with bootstrap CIs (scenario-level paired means).

Usage:
    python scripts/eval_freedrive_route.py \
        --root /home/justin_williams1/neuroncap/neuro-ncap/output/freedrive_bench \
        --tau-fail 3.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

KNOWN_CONFIGS = ["fp16", "naive_w4", "group_w4_g128", "group_w4_g64",
                 "frozen",  # optional stay-in-place control
                 "w4a4", "w4a4_dct"]


def parse_dir(name: str, configs: list[str]):
    for cfg in sorted(configs, key=len, reverse=True):
        if name.endswith("_" + cfg):
            return name[: -(len(cfg) + 1)], cfg
    return None


def _sorted_xy(ts_to_vec: dict) -> tuple[np.ndarray, np.ndarray]:
    items = sorted(((int(k), v) for k, v in ts_to_vec.items()), key=lambda kv: kv[0])
    ts = np.array([k for k, _ in items], dtype=np.float64)
    xy = np.array([[v[0], v[1]] for _, v in items], dtype=np.float64)
    return ts, xy


def _driven_xy(ego_poses: dict) -> tuple[np.ndarray, np.ndarray]:
    items = sorted(((int(k), M) for k, M in ego_poses.items()), key=lambda kv: kv[0])
    ts = np.array([k for k, _ in items], dtype=np.float64)
    xy = np.array([[M[0][3], M[1][3]] for _, M in items], dtype=np.float64)
    return ts, xy


def _point_to_polyline(p: np.ndarray, P: np.ndarray) -> float:
    """Min distance from point p to reference polyline P (segment-wise) = cross-track."""
    a, b = P[:-1], P[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab) + 1e-9
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(p - proj, axis=1)))


def route_drift(run_dir: str):
    """Return dict of route-following quantities, or None if data missing.

    cross_track : per-frame perpendicular distance to the reference PATH (lane-keeping;
                  speed-independent -- this is what corridor/off-route should use).
    progress_m  : total path length actually driven (anti-freeze: a frozen ego ~0).
    """
    try:
        ref = json.load(open(os.path.join(run_dir, "reference_trajectory.json")))
        ego = json.load(open(os.path.join(run_dir, "ego_poses.json")))
    except FileNotFoundError:
        return None
    if not ref or not ego or len(ref) < 2 or len(ego) < 1:
        return None
    _, ref_xy = _sorted_xy(ref)
    _, drv_xy = _driven_xy(ego)
    cross = np.array([_point_to_polyline(p, ref_xy) for p in drv_xy])
    progress_m = float(np.sum(np.linalg.norm(np.diff(drv_xy, axis=0), axis=1))) if len(drv_xy) > 1 else 0.0
    return {"cross_track": cross, "progress_m": progress_m, "n_frames": len(drv_xy)}


def collided(run_dir: str) -> bool | None:
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return None
    try:
        j = json.load(open(p))
    except Exception:
        return None
    c = j.get("any_collide@0.0s")
    return bool(c) if c is not None else None


def classify_collision(run_dir: str) -> str | None:
    """Classify a collision by geometry at the last ego pose.

    Returns None (no collision) or:
      'spawn' : terminated at frame <= 1 -- ego spawned overlapping an actor
                (deterministic setup artifact, e.g. scene 0099).
      'rear'  : nearest actor is >1 m BEHIND the ego (x < -1 in ego frame) --
                logged trailing traffic struck the slower-than-logged ego
                (schedule artifact, not a policy failure).
      'front' : anything else -- treated as a genuine policy collision.
      'unknown': collision flagged but geometry unavailable.
    """
    if not collided(run_dir):
        return None
    try:
        ego = json.load(open(os.path.join(run_dir, "ego_poses.json")))
        actors = json.load(open(os.path.join(run_dir, "actors.json")))
    except Exception:
        return "unknown"
    items = sorted(((int(k), np.array(M)) for k, M in ego.items()), key=lambda kv: kv[0])
    if len(items) <= 1:
        return "spawn"
    last_t, M = items[-1]
    R, t = M[:3, :3], M[:3, 3]
    best = None
    for a in actors:
        ats = np.array(a["timestamps"], dtype=float)
        ap = np.array(a["poses"])
        if ats.min() > last_t or ats.max() < last_t:
            continue
        wp = np.array([np.interp(last_t, ats, ap[:, 0, 3]),
                       np.interp(last_t, ats, ap[:, 1, 3]), t[2]])
        rel = R.T @ (wp - t)  # ego frame, +x forward
        d = float(np.hypot(rel[0], rel[1]))
        if best is None or d < best[0]:
            best = (d, rel)
    if best is None:
        return "unknown"
    return "rear" if best[1][0] < -1.0 else "front"


def eval_run(run_dir: str, tau_fail: float, min_progress: float,
             artifact_correct: bool = True):
    rd = route_drift(run_dir)
    if rd is None:
        return None
    cross = rd["cross_track"]
    ckind = classify_collision(run_dir)
    max_cross = float(np.max(cross))
    med_cross = float(np.median(cross))
    off_corridor = max_cross > tau_fail          # left the drivable corridor (lane departure)
    moved = rd["progress_m"] >= min_progress     # anti-freeze: actually drove the route
    # Artifact correction: 'spawn' and 'rear' collisions are simulation artifacts
    # (ego spawned in contact / logged traffic rear-ended the slower ego), not policy
    # failures; only 'front'/'unknown' count against success. 'spawn' runs terminate at
    # frame<=1 so the ego never got to drive -- they still fail via the progress gate,
    # and should be reported/excluded at the scene level.
    if artifact_correct:
        genuine_coll = ckind in ("front", "unknown")
    else:
        genuine_coll = ckind is not None
    # completion-style success (defn b): stayed in-corridor for the whole rollout,
    # made real forward progress (not frozen/degenerate), no genuine collision.
    success = bool((not off_corridor) and moved and not genuine_coll)
    return {
        "med_cross": med_cross, "max_cross": max_cross,
        "progress_m": rd["progress_m"], "moved": moved,
        "off_corridor": off_corridor,
        "collided": ckind is not None,
        "coll_kind": ckind or "",
        "genuine_coll": genuine_coll,
        "success": success,
    }


def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    if not values:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(values, float)
    rng = rng or np.random.default_rng(0)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--configs", nargs="*", default=KNOWN_CONFIGS)
    ap.add_argument("--tau-fail", type=float, default=4.0,
                    help="drivable-road corridor (m); max CROSS-TRACK beyond = off-route "
                         "failure. Default 4.0 = FP16's demonstrated on-road envelope "
                         "(competent-policy peak ~3.9 m to the logged path incl. maneuvers).")
    ap.add_argument("--min-progress", type=float, default=5.0,
                    help="min path length driven (m) to count as not-frozen (anti-freeze gate)")
    ap.add_argument("--no-artifact-correct", action="store_true",
                    help="count ALL collisions against success (incl. spawn/rear artifacts)")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # scenario -> config -> list of per-run metric dicts
    data = defaultdict(lambda: defaultdict(list))
    present = set()
    for d in sorted(os.listdir(args.root)):
        full = os.path.join(args.root, d)
        if not os.path.isdir(full):
            continue
        parsed = parse_dir(d, args.configs)
        if parsed is None:
            continue
        scen, cfg = parsed
        run_dirs = sorted(glob.glob(os.path.join(full, "run_*"))) or [full]
        for rdir in run_dirs:
            r = eval_run(rdir, args.tau_fail, args.min_progress,
                         artifact_correct=not args.no_artifact_correct)
            if r is not None:
                data[scen][cfg].append(r)
                present.add(cfg)
    configs = [c for c in args.configs if c in present]

    def scen_means(cfg, field):
        out = []
        for s in data:
            vals = [r[field] for r in data[s][cfg]]
            if vals:
                out.append(float(np.mean(vals)))
        return out

    corr = "OFF (all collisions count)" if args.no_artifact_correct else \
           "ON (spawn/rear collisions excluded as sim artifacts)"
    print(f"# Free-drive route competence  (root={args.root}, tau_fail={args.tau_fail} m, "
          f"min_progress={args.min_progress} m)")
    print(f"# route SUCCESS = in-corridor (cross-track) & moved & no genuine collision")
    print(f"# artifact correction: {corr}\n")
    print(f"{'config':16s} {'nruns':>6s} {'success%':>9s} {'95% CI':>16s} "
          f"{'medCross':>9s} {'progM':>7s} {'offRoute%':>10s} {'coll%':>7s} {'artif%':>7s}")
    for cfg in configs:
        n = sum(len(data[s][cfg]) for s in data)
        s_succ = scen_means(cfg, "success")
        m, lo, hi = bootstrap_ci(s_succ, args.n_boot, rng=rng)
        mc = float(np.mean([np.median([r["med_cross"] for r in data[s][cfg]])
                            for s in data if data[s][cfg]]))
        pg = float(np.mean(scen_means(cfg, "progress_m")))
        off = float(np.mean(scen_means(cfg, "off_corridor")))
        col = float(np.mean(scen_means(cfg, "genuine_coll")))
        art = float(np.mean(scen_means(cfg, "collided"))) - col
        print(f"{cfg:16s} {n:6d} {m:8.1%}   [{lo:4.1%},{hi:5.1%}] "
              f"{mc:9.2f} {pg:7.1f} {off:10.1%} {col:7.1%} {art:7.1%}")

    print(f"\n# Per-scenario route success rate")
    print("scenario".ljust(20) + "".join(f"{c:>16s}" for c in configs))
    for s in sorted(data):
        row = s.ljust(20)
        for cfg in configs:
            vals = [r["success"] for r in data[s][cfg]]
            row += (f"{np.mean(vals):16.1%}" if vals else f"{'-':>16s}")
        print(row)


if __name__ == "__main__":
    main()
