#!/usr/bin/env python3
"""Generate adversary-free ('free-drive') NeuroNCAP scenario configs.

A free-drive scenario has NO injected target actor: the ego must follow the
logged reference route through the neural renderer, closed-loop, with only the
original (reconstructed) scene actors present. This removes the stay-in-place
freeze confound of the adversarial scenarios -- a frozen ego falls off the route
and fails, so the metric measures driving *competence*, not accidental safety.

The harness supports this natively: a scenario dict with no `actors` key falls
back to the original logged actors (scenario.py), and engine.py sets
target_actor=None cleanly.

Usage:
    python scripts/gen_freedrive_scenarios.py \
        --ncap-root /home/justin_williams1/neuroncap/neuro-ncap \
        --checkpoints /home/justin_williams1/neuroncap/neurad-studio/checkpoints
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import yaml

# Prefer a frontal scenario's `general` block as the free-drive base (they tend to
# start mid-log with clean priming); fall back to any available variant.
VARIANT_PREF = ["frontal", "side", "stationary"]


def scene_ids_with_checkpoints(ckpt_dir: str) -> set[str]:
    out = set()
    for e in os.listdir(ckpt_dir):
        if re.fullmatch(r"\d{4}", e) and os.path.isdir(os.path.join(ckpt_dir, e)):
            out.add(e)
    return out


def load_general_block(ncap_root: str, scene: str) -> dict | None:
    for variant in VARIANT_PREF:
        p = os.path.join(ncap_root, "scenarios", variant, f"{scene}.yaml")
        if os.path.exists(p):
            with open(p) as f:
                d = yaml.safe_load(f)
            if d and "general" in d:
                return d["general"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncap-root", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--out-subdir", default="freedrive")
    ap.add_argument("--duration", type=float, default=None,
                    help="Override rollout duration (s). Default: keep the base "
                         "scenario's duration (guaranteed in-corridor).")
    args = ap.parse_args()

    scenes = sorted(scene_ids_with_checkpoints(args.checkpoints))
    out_dir = os.path.join(args.ncap_root, "scenarios", args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for scene in scenes:
        general = load_general_block(args.ncap_root, scene)
        if general is None:
            print(f"[skip] {scene}: no base scenario with a `general` block")
            continue
        # Free-drive: keep original start pose/velo, drop any adversarial actor.
        general = dict(general)
        general["start-pose"] = "original"
        general["start-velo"] = "original"
        if args.duration is not None:
            general["duration"] = args.duration
        # `actors: null` -> Scenario.get_actors returns the RENDERER's reconstructed actor
        # set (engine.py:301 passes renderer_actors as original_actors), i.e. the real
        # recorded traffic the NeuRAD checkpoint can render, with valid poses. No injected
        # target (target_actor=None -> reference_speed falls back to ego velocity), so it's
        # adversary-free route-following among real traffic, with a meaningful collision axis.
        # NB: `actors: []` instead triggers a NeuRAD empty-actor device bug (the baked
        # reconstructed actors get no pose update -> cuda/cpu mismatch in _get_actor_indices);
        # a missing key KeyErrors at scenario.py:57. `null` is the working, realistic choice.
        doc = {"general": general, "actors": None}
        out_path = os.path.join(out_dir, f"{scene}.yaml")
        with open(out_path, "w") as f:
            f.write("# Auto-generated free-drive (adversary-free) scenario.\n")
            f.write("# `actors: null` -> real reconstructed traffic, no injected threat.\n")
            f.write("# Ego must follow the logged reference route; measures route competence\n")
            f.write("# (frozen/degenerate policy drifts off-route or collides with real cars).\n")
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
        written.append(scene)
        print(f"[ok]   {scene} -> {out_path}  (start-frame="
              f"{general.get('start-frame')}, duration={general.get('duration')})")

    print(f"\nWrote {len(written)} free-drive scenarios to {out_dir}")
    print(f"Scenes: {' '.join(written)}")


if __name__ == "__main__":
    main()
