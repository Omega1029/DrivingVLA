"""Render the closed-loop NeuroNCAP experiments (scene-0103 frontal, seed-0) to MP4 videos from
the saved per-frame COMBINED_OUTPUTS panels (front camera + BEV + planned trajectory). No ffmpeg
needed — uses cv2.VideoWriter (mp4v). Produces per-config clips + a 3-up side-by-side."""
import glob, os
import cv2
import numpy as np

ROOT = "/home/justin_williams1/neuroncap/neuro-ncap/output/multiseed"
OUT = "/home/justin_williams1/OpenDriveVLA/videos"
os.makedirs(OUT, exist_ok=True)
VIEW = "COMBINED_OUTPUTS"
FPS = 10
HOLD = 7           # repeat each sim-frame -> ~10 s clip (14 frames * 7 / 10)

CONFIGS = [
    ("fp16",          "FP16 (baseline)  -  plans & drives, avoids",                (40, 180, 40)),
    ("naive_w4",      "naive per-channel W4  -  generation COLLAPSE, frozen -> COLLISION", (40, 40, 220)),
    ("group_w4_g128", "group-wise W4 (g128)  -  plans & drives, avoids  [THE FIX]", (40, 180, 40)),
]


def load_frames(cfg, seed=0):
    d = f"{ROOT}/{cfg}/run_{seed}/{VIEW}"
    fs = sorted(glob.glob(f"{d}/*.png")) + sorted(glob.glob(f"{d}/*.jpg"))
    return [cv2.imread(f) for f in fs]


def banner(img, text, color, h=70):
    w = img.shape[1]
    bar = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(bar, (0, 0), (w, h), (25, 25, 25), -1)
    cv2.rectangle(bar, (0, 0), (12, h), color, -1)
    scale = max(0.7, w / 1600.0)
    cv2.putText(bar, text, (28, int(h * 0.66)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def write_clip(frames, label, color, path):
    framed = [banner(f, label, color) for f in frames]
    h, w = framed[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in framed:
        for _ in range(HOLD):
            vw.write(f)
    vw.release()
    return os.path.getsize(path)


# per-config clips
clips = {}
for cfg, label, color in CONFIGS:
    fr = load_frames(cfg)
    if not fr:
        print(f"!! no frames for {cfg}"); continue
    clips[cfg] = fr
    p = f"{OUT}/0103_seed0_{cfg}.mp4"
    sz = write_clip(fr, label, color, p)
    print(f"wrote {p}  ({len(fr)} frames, {sz//1024} KB)")

# 3-up side-by-side (scaled to common height)
if len(clips) == 3:
    TH = 300
    def scaled(cfg):
        out = []
        for f in clips[cfg]:
            s = TH / f.shape[0]
            out.append(cv2.resize(f, (int(f.shape[1] * s), TH)))
        return out
    cols = {c: scaled(c) for c, _, _ in CONFIGS}
    n = min(len(v) for v in cols.values())
    labels = {c: (l, col) for c, l, col in CONFIGS}
    rows = []
    for i in range(n):
        panels = []
        for c, _, _ in CONFIGS:
            short = {"fp16": "FP16", "naive_w4": "naive W4 (COLLAPSE)",
                     "group_w4_g128": "group W4 (FIX)"}[c]
            panels.append(banner(cols[c][i], short, labels[c][1], h=44))
        rows.append(np.hstack(panels))
    h, w = rows[0].shape[:2]
    p = f"{OUT}/0103_seed0_sidebyside.mp4"
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in rows:
        for _ in range(HOLD):
            vw.write(f)
    vw.release()
    print(f"wrote {p}  ({n} frames, {os.path.getsize(p)//1024} KB, {w}x{h})")
print("DONE")
