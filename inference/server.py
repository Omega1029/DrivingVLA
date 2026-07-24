"""NeuroNCAP model-node server for OpenDriveVLA.

Implements the endpoints the NeuroNCAP orchestrator calls (mirrors the UniAD example):
  GET  /alive  -> bool
  POST /infer  -> {trajectory: [[x,y], ...], aux_outputs: {}}  (ego-frame BEV, 6 waypoints @ 2 Hz)
  POST /reset  -> bool                                          (clear per-scenario state)

The orchestrator (neuro_ncap/components/model_api.py::ModelInput.to_json_dict) sends:
  images: {cam -> base64 of torch.save(uint8 HWC RGB tensor)}, ego2world 4x4, canbus 16-vec,
  timestamp (us), command (int), calibration{camera2image, camera2ego, lidar2ego}.

Run natively (no Docker):
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=<gpu> venv/bin/python -m uvicorn inference.server:app \
      --host 0.0.0.0 --port 9000
Env: ODV_CKPT=checkpoints/OpenDriveVLA-0.5B  ODV_WQUANT_BITS=0  (4 for the W4 run)
     ODV_DEVICE=cuda:0  ODV_AUTOCAST=float16
"""
import base64
import io
import os
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from inference.runner import OpenDriveVLARunner, InferenceInput

NUSCENES_CAM_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


class Calibration(BaseModel):
    camera2image: Dict[str, List[List[float]]]
    camera2ego: Dict[str, List[List[float]]]
    lidar2ego: List[List[float]]


class InferenceInputs(BaseModel):
    images: Dict[str, str]              # cam -> base64(torch.save(uint8 HWC RGB tensor))
    ego2world: List[List[float]]        # 4x4
    canbus: List[float]                 # 16-vec emulated by the orchestrator
    timestamp: int                      # microseconds
    command: int                        # 0 right / 1 left / 2 straight
    calibration: Calibration


class InferenceOutputs(BaseModel):
    trajectory: List[List[float]]       # [[x,y], ...] ego-frame BEV
    aux_outputs: dict = {}


app = FastAPI()
_runner: OpenDriveVLARunner | None = None


def _get_runner() -> OpenDriveVLARunner:
    global _runner
    if _runner is None:
        _runner = OpenDriveVLARunner(
            ckpt_path=os.environ.get("ODV_CKPT", "checkpoints/OpenDriveVLA-0.5B"),
            wquant_bits=int(os.environ.get("ODV_WQUANT_BITS", "0")),
            device=os.environ.get("ODV_DEVICE", "cuda:0"),
            autocast_dtype=os.environ.get("ODV_AUTOCAST", "float16"),
        )
        # Auto-load calibrated AWQ scales so mode="awq" works on a fresh node
        # (scales were calibrated once on rendered-domain activations; see run_awq.sh).
        awq_path = os.environ.get("ODV_AWQ_SCALES", "logs/awq_scales.pt")
        if os.path.exists(awq_path):
            _runner.load_awq_scales(awq_path)
            print(f">>> loaded AWQ scales from {awq_path}")
    return _runner


def _b64_tensor_to_rgb(b64: str) -> np.ndarray:
    """Decode base64(torch.save(uint8 HWC RGB tensor)) -> (H,W,3) uint8 RGB ndarray."""
    raw = base64.b64decode(b64)
    t = torch.load(io.BytesIO(raw), map_location="cpu")
    return t.numpy().astype(np.uint8)


@app.get("/alive")
async def alive() -> bool:
    return True


@app.post("/reset")
async def reset_runner() -> bool:
    _get_runner().reset()
    return True


class QuantRequest(BaseModel):
    bits: int
    group: int = 0
    mode: str = "rtn_sym"   # rtn_sym | rtn_asym | awq | fp16
    a_bits: int = 0         # activation fake-quant bits (0 = off)
    rot: str = ""           # rotation kind: dct | hadamard | random_orthogonal | none | ""


@app.post("/quantize")
async def quantize(req: QuantRequest) -> dict:
    """Re-quantize the backbone in-place from the FP snapshot (sweep techniques without reload).
    If a_bits or rot is set, uses the module-wrapping rotation/activation path (rotquant)."""
    return _get_runner().apply_quant(bits=req.bits, group=req.group, mode=req.mode,
                                     a_bits=req.a_bits, rot=req.rot)


@app.get("/quant_cfg")
async def quant_cfg() -> dict:
    return _get_runner().quant_cfg


@app.post("/awq_calib_start")
async def awq_calib_start() -> bool:
    _get_runner().awq_calib_start()
    return True


@app.post("/awq_calib_finish")
async def awq_calib_finish(alpha: float = 0.5) -> dict:
    r = _get_runner()
    out = r.awq_calib_finish(alpha=alpha)
    r.save_awq_scales(os.environ.get("ODV_AWQ_SCALES", "logs/awq_scales.pt"))
    return out


@app.post("/infer")
async def infer(data: InferenceInputs) -> InferenceOutputs:
    imgs = np.stack([_b64_tensor_to_rgb(data.images[c]) for c in NUSCENES_CAM_ORDER], axis=0)
    out = _get_runner().run(InferenceInput(
        imgs=imgs,
        ego2world=np.asarray(data.ego2world, dtype=np.float64),
        can_bus=np.asarray(data.canbus, dtype=np.float64),
        command=int(data.command),
        timestamp=data.timestamp / 1e6,  # -> seconds
        camera2image={k: np.asarray(v) for k, v in data.calibration.camera2image.items()},
        camera2ego={k: np.asarray(v) for k, v in data.calibration.camera2ego.items()},
        lidar2ego=np.asarray(data.calibration.lidar2ego),
        cam_order=NUSCENES_CAM_ORDER,
    ))
    return InferenceOutputs(trajectory=out.trajectory.tolist(), aux_outputs=out.aux_outputs)
