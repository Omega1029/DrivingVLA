"""Synthesize a nuScenes CAN-bus 'pose' + 'zoe_veh_info' file for scene-0103 from the (accurate)
LIDAR_TOP ego-pose track, so NeuroNCAP can run without the gated can_bus.zip expansion.

The orchestrator only consumes 'pose' (pos, orientation, accel, rotation_rate, vel) and
'zoe_veh_info' (steer_corrected), and only for priming/initial state — afterwards it emulates
can_bus from the driven poses. Velocities/accels here are finite-differenced from the dense
(~20 Hz) lidar ego-pose track and expressed in the ego frame, matching the real CAN layout.
"""
import json
import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw

DATAROOT = "/home/justin_williams1/OpenDriveVLA/data/nuscenes"
SCENE = "scene-0103"

nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
scene = next(s for s in nusc.scene if s["name"] == SCENE)
first_sample = nusc.get("sample", scene["first_sample_token"])

# Walk the LIDAR_TOP sample_data chain (keyframes + sweeps) for a dense, single-sensor pose track.
sd = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
poses = []
while True:
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    poses.append((ego["timestamp"], np.array(ego["translation"], float),
                  np.array(ego["rotation"], float)))
    if sd["next"] == "":
        break
    sd = nusc.get("sample_data", sd["next"])

# Sort + dedupe by timestamp.
poses.sort(key=lambda p: p[0])
uniq = []
for p in poses:
    if not uniq or p[0] > uniq[-1][0]:
        uniq.append(p)
poses = uniq
n = len(poses)
print(f"{SCENE}: {n} ego poses, span {(poses[-1][0]-poses[0][0])/1e6:.1f}s")

times = np.array([p[0] for p in poses], float)            # us
pos = np.stack([p[1] for p in poses])                     # (n,3) global
quats = [Quaternion(p[2]) for p in poses]
yaws = np.array([quaternion_yaw(q) for q in quats])

# Global velocity via central differences, then rotate into the (yaw-only) ego frame.
dt = np.gradient(times) / 1e6                              # s
vel_global = np.stack([np.gradient(pos[:, i], times / 1e6) for i in range(3)], axis=1)
yaw_rate = np.gradient(np.unwrap(yaws), times / 1e6)

pose_msgs = []
vel_ego_prev = None
ego_vels = []
for i in range(n):
    Rz = Quaternion(axis=[0, 0, 1], radians=yaws[i]).rotation_matrix
    v_ego = Rz.T @ vel_global[i]
    ego_vels.append(v_ego)
ego_vels = np.stack(ego_vels)
accel_ego = np.stack([np.gradient(ego_vels[:, i], times / 1e6) for i in range(3)], axis=1)

for i in range(n):
    pose_msgs.append({
        "utime": int(times[i]),
        "pos": pos[i].tolist(),
        "orientation": list(quats[i].elements),                 # [w,x,y,z]
        "accel": [float(accel_ego[i, 0]), float(accel_ego[i, 1]), 9.81],
        "rotation_rate": [0.0, 0.0, float(yaw_rate[i])],
        "vel": [float(ego_vels[i, 0]), float(ego_vels[i, 1]), 0.0],
    })

# zoe_veh_info: steering. Approximate steer (rad at wheel) ~ atan(yaw_rate*wheelbase/speed),
# stored as steer_corrected in DEGREES at the wheel-times-ratio (reader multiplies by pi/180).
WHEELBASE = 2.588
STEER_RATIO = 16.0
veh_msgs = []
for i in range(n):
    speed = float(np.hypot(ego_vels[i, 0], ego_vels[i, 1]))
    wheel = np.arctan(yaw_rate[i] * WHEELBASE / speed) if speed > 0.5 else 0.0
    veh_msgs.append({"utime": int(times[i]), "steer_corrected": float(np.degrees(wheel) * STEER_RATIO)})

out_dir = os.path.join(DATAROOT, "can_bus")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f"{SCENE}_pose.json"), "w") as f:
    json.dump(pose_msgs, f)
with open(os.path.join(out_dir, f"{SCENE}_zoe_veh_info.json"), "w") as f:
    json.dump(veh_msgs, f)

print("wrote:", os.path.join(out_dir, f"{SCENE}_pose.json"))
print("wrote:", os.path.join(out_dir, f"{SCENE}_zoe_veh_info.json"))
print(f"sample mid-scene vel(ego)={ego_vels[n//2]}, yaw_rate={yaw_rate[n//2]:.4f}, "
      f"speed={np.hypot(*ego_vels[n//2][:2]):.2f} m/s")
