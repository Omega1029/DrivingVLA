#!/usr/bin/env python
"""Compute open-loop L2 (ST-P3 style) over the captured subset for one or more plan_conv files,
and pairwise trajectory agreement between files. GT from drivevla/eval_share/gt/gt_traj.pkl."""
import json, pickle, sys, os, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drivevla"))
from utils.trajectory_utils import retrieve_traj

GTD = "drivevla/eval_share/gt"
gt = pickle.load(open(f"{GTD}/gt_traj.pkl", "rb"))
gm = pickle.load(open(f"{GTD}/gt_traj_mask.pkl", "rb"))

def load_preds(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        tok = r["id"].split("_")[0]
        a = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
        try:
            t = np.array(retrieve_traj(a), dtype=np.float64).reshape(6, 2)
        except Exception:
            t = np.full((6, 2), np.nan)
        out[tok] = t
    return out

def l2_vs_gt(preds):
    # ST-P3: cumulative-average L2 at 1s/2s/3s (waypoints at 0.5s steps: idx 1,3,5)
    per = []
    for tok, p in preds.items():
        if tok not in gt:
            continue
        g = np.array(gt[tok]).reshape(6, 2)
        m = np.array(gm[tok]).reshape(6, 2)[:, 0] > 0
        d = np.linalg.norm(p - g, axis=1)
        per.append(np.where(m, d, np.nan))
    per = np.array(per)  # (N,6)
    n = per.shape[0]
    def cumavg(k):  # mean over steps 0..k, then over samples
        return np.nanmean(np.nanmean(per[:, :k+1], axis=1))
    return n, cumavg(1), cumavg(3), cumavg(5), np.nanmean(per)

def agree(a, b):
    toks = set(a) & set(b)
    ds = [np.linalg.norm(a[t] - b[t], axis=1).mean() for t in toks
          if not (np.isnan(a[t]).any() or np.isnan(b[t]).any())]
    return len(ds), float(np.mean(ds)), float(np.max(ds))

if __name__ == "__main__":
    files = sys.argv[1:]
    print(f"{'file':40s} {'N':>3} {'L2@1s':>7} {'L2@2s':>7} {'L2@3s':>7} {'L2avg':>7}")
    P = {}
    for f in files:
        p = load_preds(f); P[f] = p
        n, a, b, c, m = l2_vs_gt(p)
        print(f"{os.path.basename(f):40s} {n:3d} {a:7.3f} {b:7.3f} {c:7.3f} {m:7.3f}")
