"""OpenDriveVLA runner for NeuroNCAP closed-loop.

Turns one rendered frame (6 cam images + ego pose + calibration + emulated can_bus + command)
into a planned trajectory. OpenDriveVLA perception IS UniAD stage-1 track+map, so the
image/pose -> perception preprocessing follows the OpenDriveVLA dataset pipeline (which mirrors
UniAD), verified against:
  - projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py  (can_bus fill: +rotation, patch_angle)
  - projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py::forward_test
      img -> [tensor], img_metas -> [[dict]], timestamp -> [ts]; the detector applies the
      can_bus temporal delta itself (prev_pos/prev_angle), so we pass ABSOLUTE can_bus.
  - drivevla/data_utils/build_llava_conversation.py  (exact ego-state / mission-goal prompt text)

The orchestrator (neuro-ncap) sends a 16-vec emulated can_bus and a Command int (0 right,
1 left, 2 straight); we use those rather than re-deriving, matching how training built them.
"""
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

from drivevla.utils.trajectory_utils import retrieve_traj

# nuScenes ResNet-caffe norm from base_track_map img_norm_cfg (BGR, to_rgb=False)
IMG_MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)
IMG_STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)
SIZE_DIVISOR = 32

# Command int -> mission-goal text (matches build_llava_conversation.py; orchestrator Command enum
# RIGHT=0, LEFT=1, STRAIGHT=2).
COMMAND_TEXT = {0: "turn right", 1: "turn left", 2: "keep forward"}


@dataclass
class InferenceInput:
    imgs: np.ndarray                 # (6, H, W, 3) RGB uint8, NUSCENES_CAM_ORDER
    ego2world: np.ndarray            # (4, 4)
    can_bus: np.ndarray              # (16,) emulated by the orchestrator
    command: int                     # 0 right / 1 left / 2 straight
    timestamp: float                 # seconds
    camera2image: Dict[str, np.ndarray]
    camera2ego: Dict[str, np.ndarray]
    lidar2ego: np.ndarray
    cam_order: List[str]


@dataclass
class InferenceOutput:
    trajectory: np.ndarray           # (6, 2) ego-frame BEV @ 2 Hz
    aux_outputs: dict = field(default_factory=dict)


def _normalize_pad(imgs_rgb: np.ndarray) -> tuple[torch.Tensor, tuple]:
    """RGB uint8 (N,H,W,3) -> normalized, padded-to-/32, (1,N,3,H',W') float32 tensor."""
    imgs = imgs_rgb[..., ::-1].astype(np.float32)          # RGB->BGR (to_rgb=False)
    imgs = (imgs - IMG_MEAN) / IMG_STD                     # NormalizeMultiviewImage
    n, h, w, c = imgs.shape
    ph = int(np.ceil(h / SIZE_DIVISOR) * SIZE_DIVISOR)
    pw = int(np.ceil(w / SIZE_DIVISOR) * SIZE_DIVISOR)
    padded = np.zeros((n, ph, pw, c), dtype=np.float32)    # PadMultiViewImage (pad value 0)
    padded[:, :h, :w, :] = imgs
    t = torch.from_numpy(np.ascontiguousarray(padded.transpose(0, 3, 1, 2)))  # N,C,H,W
    return t.unsqueeze(0), (ph, pw, c)                     # 1,N,C,H,W


def _to4x4(k: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    k = np.asarray(k)
    m[:k.shape[0], :k.shape[1]] = k
    return m


class OpenDriveVLARunner:
    def __init__(self, ckpt_path: str, wquant_bits: int = 0, device: str = "cuda:0",
                 autocast_dtype: str = "float16"):
        self.device = device
        self.wquant_bits = wquant_bits
        self.autocast_dtype = torch.bfloat16 if autocast_dtype == "bfloat16" else torch.float16
        self._load_model(ckpt_path, wquant_bits)
        self.reset()

    def _load_model(self, ckpt_path, wquant_bits):
        import os as _os, sys as _sys
        # inference_drivevla.py imports `data_utils.*` (a drivevla/-relative package), so the
        # drivevla/ dir must be on sys.path for that import to resolve.
        _dv = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "drivevla")
        if _dv not in _sys.path:
            _sys.path.insert(0, _dv)
        from llava.model.builder import load_pretrained_model
        from inference.quantizers import QuantState
        overwrite = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            ckpt_path, model_base=None, model_name="llava_qwen",
            device_map=self.device, multimodal=True,
            attn_implementation="sdpa", overwrite_config=overwrite,
        )
        self.model.eval()
        # Snapshot FP weights once so quant techniques can be swept in-place (no reload).
        self.qstate = QuantState(self.model)
        self.quant_cfg = {"mode": "fp16", "bits": 0, "group": 0}
        if wquant_bits and wquant_bits > 0:
            self.apply_quant(bits=wquant_bits,
                             group=int(__import__("os").environ.get("ODV_WQUANT_GROUP", "0")),
                             mode=__import__("os").environ.get("ODV_WQUANT_MODE", "rtn_sym"))

    def apply_quant(self, bits: int, group: int = 0, mode: str = "rtn_sym",
                    a_bits: int = 0, rot: str = "") -> dict:
        """Re-quantize the backbone in-place from the FP snapshot. Returns a summary dict.

        Two paths:
        - weight-only (default): QuantState in-place RTN/AWQ on the original Linear weights.
        - rotation / activation-quant (a_bits or rot set): module-wrapping via rotquant
          (y = quant_act(x Q^T) @ quant_w(W Q^T)^T). Always restores FP + unwraps first,
          so paths compose cleanly across successive /quantize calls.
        """
        from inference.rotquant import apply_rotated, remove_rotated
        # Always return to a clean FP, unwrapped state first.
        n_unwrapped = remove_rotated(self.model)
        self.qstate.restore_fp()
        if a_bits or rot:
            summary = apply_rotated(self.model, w_bits=bits or 16, a_bits=a_bits or 16,
                                    kind=rot or "none", group=group)
            summary["unwrapped_prev"] = n_unwrapped
            self.quant_cfg = {"mode": f"rot_{rot or 'none'}", "bits": bits,
                              "group": group, "a_bits": a_bits}
        else:
            summary = self.qstate.apply(bits=bits, group=group, mode=mode)
            self.quant_cfg = {"mode": mode, "bits": bits, "group": group}
        print(f">>> apply_quant: {summary}")
        return summary

    def awq_calib_start(self):
        """Capture backbone activations during subsequent /infer calls (run on the FP model)."""
        self.qstate.restore_fp()
        self.qstate.calib_start()
        print(">>> AWQ calibration: capturing activations")

    def awq_calib_finish(self, alpha: float = 0.5) -> dict:
        out = self.qstate.calib_finish(alpha=alpha)
        print(f">>> AWQ calibration done: {out}")
        return out

    def save_awq_scales(self, path: str):
        torch.save(self.qstate.awq_scales, path)

    def load_awq_scales(self, path: str):
        self.qstate.awq_scales = torch.load(path, map_location=self.device)

    def _detector(self):
        """Best-effort handle on the mmdet3d UniAD detector (for prev_frame_info reset)."""
        try:
            vt = self.model.get_vision_tower()
            return vt.vision_tower.vision_model
        except Exception:
            return None

    def reset(self):
        # New scene token per scenario so the detector drops temporal BEV + resets pose deltas.
        self.scene_token = str(uuid.uuid4())
        det = self._detector()
        if det is not None and hasattr(det, "prev_frame_info"):
            det.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                   "prev_pos": 0, "prev_angle": 0}

    # ── ego-state text fields, read from the orchestrator's emulated can_bus ──
    # can_bus layout (emulate_nuscenes_canbus_signals): [0:3] global trans, [3:7] yaw quat,
    # [7:10] accel ego-frame, [10:13] rot-rate ego-frame (yaw only), [13:16] vel ego-frame.
    def _ego_state_text(self, can_bus, command):
        vx, vy = can_bus[13] * 0.5, can_bus[14] * 0.5      # *0.5 matches training prompt scaling
        v_yaw = can_bus[12]
        ax, ay = can_bus[7], can_bus[8]
        cx, cy = can_bus[7], can_bus[8]                    # "Can Bus" accel field
        vhead = float(np.hypot(can_bus[13], can_bus[14])) * 0.5
        ego = (f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f})"
               f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})"
               f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
               f" - Can Bus: ({cx:.2f},{cy:.2f})"
               f" - Heading Speed: ({vhead:.2f})"
               f" - Steering: (0.00)")
        return ego, COMMAND_TEXT.get(int(command), "keep forward")

    # ── perception input dict (mirrors the dataset pipeline + forward_test contract) ──
    def _build_perception_input(self, inp: InferenceInput):
        img_t, (ph, pw, _) = _normalize_pad(inp.imgs)
        lidar2ego = _to4x4(inp.lidar2ego)
        lidar2img = []
        for cam in inp.cam_order:
            intr = _to4x4(inp.camera2image[cam])
            lidar2cam = np.linalg.inv(_to4x4(inp.camera2ego[cam])) @ lidar2ego
            lidar2img.append(intr @ lidar2cam)
        lidar2img = np.stack(lidar2img, axis=0)

        lidar2world = inp.ego2world @ lidar2ego
        l2g_t = torch.tensor(lidar2world[:3, 3], dtype=torch.float32, device=self.device).unsqueeze(0)
        l2g_r = torch.tensor(lidar2world[:3, :3], dtype=torch.float32, device=self.device).unsqueeze(0)

        # can_bus (16 -> 18): append patch_angle (rad, deg); set [3:7]=rotation per training.
        # ABSOLUTE values — forward_test computes the temporal delta itself.
        can_bus = np.array(inp.can_bus, dtype=np.float64).copy()
        rotation = Quaternion(can_bus[3:7])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus = np.append(can_bus, patch_angle / 180 * np.pi)   # idx 16
        can_bus = np.append(can_bus, patch_angle)                 # idx 17
        can_bus[3:7] = rotation.elements                          # +rotation (training convention)

        from mmdet3d.core.bbox import DepthInstance3DBoxes
        h0, w0 = inp.imgs.shape[1], inp.imgs.shape[2]
        img_meta = {
            "scene_token": self.scene_token,
            "can_bus": can_bus,
            "lidar2img": list(lidar2img),
            "img_shape": [(ph, pw, 3)] * len(inp.cam_order),
            "ori_shape": (h0, w0, 3),
            "pts_filename": f"{self.scene_token}.bin",            # panseg get_bboxes splits this
            "sample_idx": self.scene_token,                       # used as result['token']
            "box_type_3d": DepthInstance3DBoxes,
        }
        # sdc_planning / command are passed to get_results_for_vlm; unused in test mode but must
        # be indexable ([0]). Heads that would consume them (motion/occ/planning) are disabled.
        return {
            "img": [img_t.to(self.device)],                       # forward_test does img[0]
            "img_metas": [[img_meta]],                            # forward_test does img_metas[0][0]
            "l2g_t": l2g_t,
            "l2g_r_mat": l2g_r,
            "timestamp": torch.tensor([[inp.timestamp]], dtype=torch.float64, device=self.device),
            "command": torch.tensor([int(inp.command)], device=self.device),
            "sdc_planning": torch.zeros(1, 1, 6, 3, device=self.device),
            "sdc_planning_mask": torch.zeros(1, 1, 6, 3, device=self.device),
        }

    def _build_prompt(self, ego_text, mission_text):
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_uniad_token
        from drivevla.data_utils.build_llava_conversation import (
            DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_TOKEN, DEFAULT_SCENE_END_TOKEN,
            DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_TOKEN, DEFAULT_TRACK_END_TOKEN,
            DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_TOKEN, DEFAULT_MAP_END_TOKEN, DEFAULT_TRAJ_TOKEN)
        question = (
            f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
            f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
            f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
            f"Ego states: {ego_text}\n"
            f"Historical trajectory (last 2 seconds): [(0.00,0.00),(0.00,0.00),(0.00,0.00),(0.00,0.00)]\n"
            f"Mission goal: {mission_text}\n"
            f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}")
        conv = conv_templates["qwen_planning_oriented_vlm"].copy()
        conv.clear_conversation()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return tokenizer_uniad_token(prompt, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def run(self, inp: InferenceInput) -> InferenceOutput:
        ego_text, mission_text = self._ego_state_text(np.asarray(inp.can_bus), inp.command)
        uniad_data = self._build_perception_input(inp)
        input_ids = self._build_prompt(ego_text, mission_text)
        with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
            cont = self.model.generate(
                input_ids, uniad_data=uniad_data, uniad_pth=None,
                do_sample=False, temperature=0, max_new_tokens=512, num_beams=1)
        answer = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        traj = np.asarray(retrieve_traj(answer), dtype=np.float64).reshape(6, 2)
        return InferenceOutput(trajectory=traj, aux_outputs={})
