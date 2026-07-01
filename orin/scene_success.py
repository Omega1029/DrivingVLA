#!/usr/bin/env python
"""Per-scene open-loop success breakdown for plan_conv files.
Groups samples by nuScenes scene and reports N, mean L2 @3s, and success rate
(% of predictions within a distance of the human GT trajectory)."""
import json, pickle, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drivevla"))
from utils.trajectory_utils import retrieve_traj

GTD = "drivevla/eval_share/gt"
gt = pickle.load(open(f"{GTD}/gt_traj.pkl", "rb"))
gm = pickle.load(open(f"{GTD}/gt_traj_mask.pkl", "rb"))
info = pickle.load(open("data/infos/nuscenes_infos_temporal_val_mini.pkl", "rb"))
infos = info["infos"] if isinstance(info, dict) else info
tok2scene = {s["token"]: s["scene_token"] for s in infos}
name = {}
for sj in ["data/nuscenes/v1.0-mini/scene.json", "data/nuscenes/v1.0-trainval/scene.json"]:
    if os.path.exists(sj):
        for r in json.load(open(sj)):
            name[r["token"]] = r["name"]

def per_sample(path):
    rows = {}
    for l in open(path):
        r = json.loads(l); tok = r["id"].split("_")[0]
        a = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
        try:
            p = np.array(retrieve_traj(a), dtype=np.float64).reshape(6, 2)
        except Exception:
            p = np.full((6, 2), np.nan)
        if tok not in gt:
            continue
        g = np.array(gt[tok]).reshape(6, 2); m = np.array(gm[tok]).reshape(6, 2)[:, 0] > 0
        d = np.linalg.norm(p - g, axis=1)
        rows[tok] = {"meanL2": float(np.nanmean(np.where(m, d, np.nan))) if not np.isnan(p).any() else np.inf,
                     "l2_3s": float(d[5]) if (m[5] and not np.isnan(p).any()) else np.inf,
                     "malformed": bool(np.isnan(p).any())}
    return rows

def report(path, thr=(1.0, 1.5, 2.0)):
    rows = per_sample(path)
    byscene = {}
    for tok, r in rows.items():
        sc = name.get(tok2scene.get(tok, "?"), tok2scene.get(tok, "?"))
        byscene.setdefault(sc, []).append(r)
    print(f"\n== {os.path.basename(path)} ==")
    hdr = f"{'scene':14s} {'N':>3} {'meanL2':>7} {'L2@3s':>7} " + " ".join(f'<={t}m'.rjust(7) for t in thr) + f" {'malf':>5}"
    print(hdr)
    allm = []
    for sc in sorted(byscene):
        rs = byscene[sc]; N = len(rs)
        mean = np.mean([x["meanL2"] for x in rs if np.isfinite(x["meanL2"])]) if any(np.isfinite(x["meanL2"]) for x in rs) else float('nan')
        l3 = np.mean([x["l2_3s"] for x in rs if np.isfinite(x["l2_3s"])]) if any(np.isfinite(x["l2_3s"]) for x in rs) else float('nan')
        succ = [100*np.mean([x["meanL2"] <= t for x in rs]) for t in thr]
        malf = sum(x["malformed"] for x in rs)
        allm += rs
        print(f"{sc:14s} {N:3d} {mean:7.3f} {l3:7.3f} " + " ".join(f"{s:6.1f}%" for s in succ) + f" {malf:5d}")
    N = len(allm)
    mean = np.mean([x["meanL2"] for x in allm if np.isfinite(x["meanL2"])])
    l3 = np.mean([x["l2_3s"] for x in allm if np.isfinite(x["l2_3s"])])
    succ = [100*np.mean([x["meanL2"] <= t for x in allm]) for t in thr]
    print(f"{'ALL':14s} {N:3d} {mean:7.3f} {l3:7.3f} " + " ".join(f"{s:6.1f}%" for s in succ) + f" {sum(x['malformed'] for x in allm):5d}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
