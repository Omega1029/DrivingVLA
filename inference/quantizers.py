"""Weight-only PTQ techniques for the OpenDriveVLA Qwen backbone, for the closed-loop
quantization study. All methods are *fake* quant (quantize→dequantize to the original dtype),
applied only to the nn.Linear weights of the Qwen decoder stack — the UniAD perception tower,
the scene/track/map projectors, embeddings and lm_head are left in FP, so the measured effect is
purely the language/action head's W{bits} loss.

Techniques (mode):
  rtn_sym   : per-output-channel (or per-group) symmetric round-to-nearest  [baseline]
  rtn_asym  : asymmetric RTN with a zero-point (better for non-zero-centred weights)
  awq       : activation-aware weight quant — per-input-channel scaling s from calibration,
              folded as W_q = Q(W·diag(s))·diag(1/s) so salient (high-activation) channels
              quantize more accurately. Pure weight transform (no runtime activation scaling).

`group` = 0 → per-output-channel; >0 → group size along the input dim (e.g. 128).

`QuantState` snapshots the pristine FP weights once so techniques can be swept in-place without
reloading the model (re-quantize from FP each time).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _iter_backbone_linears(model):
    """Yield the nn.Linear modules of the Qwen decoder stack (model.model.layers)."""
    backbone = model
    for _ in range(3):
        if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
            backbone = backbone.model
            break
        if hasattr(backbone, "layers"):
            break
        backbone = getattr(backbone, "model", backbone)
    layers = getattr(backbone, "layers", None)
    assert layers is not None, "could not locate Qwen decoder layers"
    for m in layers.modules():
        if isinstance(m, nn.Linear):
            yield m


def _quantize_weight(wf: torch.Tensor, bits: int, group: int, asym: bool) -> torch.Tensor:
    """Quantize one FP32 weight (out,in) per-output-channel or per-group; sym or asym RTN."""
    out_f, in_f = wf.shape
    if group and group > 0 and in_f % group == 0:
        w = wf.reshape(out_f, in_f // group, group)
        red = -1
    else:
        w = wf
        red = 1
    if asym:
        qmax = 2 ** bits - 1
        wmin = w.amin(dim=red, keepdim=True)
        wmax = w.amax(dim=red, keepdim=True)
        scale = (wmax - wmin).clamp_(min=1e-8) / qmax
        zp = torch.round(-wmin / scale)
        wq = (torch.clamp(torch.round(w / scale) + zp, 0, qmax) - zp) * scale
    else:
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
        scale = w.abs().amax(dim=red, keepdim=True).clamp_(min=1e-8) / qmax
        wq = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    return wq.reshape(out_f, in_f)


class QuantState:
    """Holds the pristine FP weights so techniques can be applied/swept in-place."""

    def __init__(self, model):
        self.model = model
        self.linears = list(_iter_backbone_linears(model))
        # snapshot FP weights (cpu to save GPU mem; small for 0.5B)
        self._fp = [m.weight.data.detach().clone() for m in self.linears]
        self.n_params = sum(w.numel() for w in self._fp)
        self.awq_scales: list[torch.Tensor] | None = None  # per-linear per-input-channel scale

    def restore_fp(self):
        for m, w in zip(self.linears, self._fp):
            m.weight.data.copy_(w)

    @torch.no_grad()
    def apply(self, bits: int, group: int = 0, mode: str = "rtn_sym") -> dict:
        """Restore FP then apply the chosen technique in-place. Returns a small summary."""
        self.restore_fp()
        if bits <= 0 or bits >= 16:
            return {"mode": "fp16", "bits": bits, "group": group, "n_linear": len(self.linears),
                    "n_params": self.n_params}
        asym = mode == "rtn_asym"
        use_awq = mode == "awq"
        if use_awq and self.awq_scales is None:
            raise RuntimeError("awq mode requires calibration first (awq_scales is None)")
        for i, m in enumerate(self.linears):
            wf = m.weight.data.float()
            if use_awq:
                s = self.awq_scales[i].to(wf.device, wf.dtype).clamp_(min=1e-4)  # (in_f,)
                wq = _quantize_weight(wf * s.unsqueeze(0), bits, group, asym=True) / s.unsqueeze(0)
            else:
                wq = _quantize_weight(wf, bits, group, asym)
            m.weight.data.copy_(wq.to(m.weight.dtype))
        return {"mode": mode, "bits": bits, "group": group,
                "n_linear": len(self.linears), "n_params": self.n_params,
                "awq": use_awq}

    # ── streaming AWQ calibration (capture across live /infer calls) ──
    def calib_start(self):
        """Begin accumulating per-Linear input |activation| via forward hooks."""
        self._calib_sums = [torch.zeros(m.weight.shape[1], device=m.weight.device,
                                        dtype=torch.float32) for m in self.linears]
        self._calib_counts = [0 for _ in self.linears]
        idx_of = {id(m): i for i, m in enumerate(self.linears)}
        self._calib_handles = []

        def mk_hook(mod):
            i = idx_of[id(mod)]
            def hook(_m, inp, _out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).abs().float().sum(0)
                self._calib_sums[i] += x
                self._calib_counts[i] += 1
            return hook

        for m in self.linears:
            self._calib_handles.append(m.register_forward_hook(mk_hook(m)))

    @torch.no_grad()
    def calib_finish(self, alpha: float = 0.5) -> dict:
        """Stop capturing, compute per-input-channel AWQ scales s = mean(|x|)^alpha (unit mean)."""
        for h in getattr(self, "_calib_handles", []):
            h.remove()
        self._calib_handles = []
        scales = []
        for i in range(len(self.linears)):
            mean_abs = (self._calib_sums[i] / max(self._calib_counts[i], 1)).clamp_(min=1e-6)
            s = mean_abs ** alpha
            s = s / s.mean().clamp_(min=1e-8)
            scales.append(s)
        self.awq_scales = scales
        return {"n_linear": len(self.linears), "alpha": alpha,
                "calib_batches": int(sum(self._calib_counts) / max(len(self._calib_counts), 1))}

    # ── AWQ calibration: per-input-channel activation scale s = mean(|x|)^alpha ──
    @torch.no_grad()
    def calibrate_awq(self, run_forward, alpha: float = 0.5):
        """Collect per-Linear input activation magnitude by hooking the backbone Linears while
        `run_forward()` does one or more representative forward passes, then set awq_scales.
        s_j = (mean_t |x_{t,j}|)^alpha, normalized to unit mean per layer."""
        sums = [torch.zeros(m.weight.shape[1], device=m.weight.device, dtype=torch.float32)
                for m in self.linears]
        counts = [0 for _ in self.linears]
        idx_of = {id(m): i for i, m in enumerate(self.linears)}
        handles = []

        def mk_hook(mod):
            i = idx_of[id(mod)]
            def hook(_m, inp, _out):
                x = inp[0]
                x = x.reshape(-1, x.shape[-1]).abs().float().sum(0)
                sums[i] += x
                counts[i] += 1
            return hook

        for m in self.linears:
            handles.append(m.register_forward_hook(mk_hook(m)))
        try:
            run_forward()
        finally:
            for h in handles:
                h.remove()

        scales = []
        for i in range(len(self.linears)):
            mean_abs = sums[i] / max(counts[i], 1)
            s = mean_abs.clamp_(min=1e-6) ** alpha
            s = s / s.mean().clamp_(min=1e-8)   # normalize → unit mean (keeps weight magnitude sane)
            scales.append(s)
        self.awq_scales = scales
        return {"n_linear": len(self.linears), "alpha": alpha,
                "calib_batches": int(sum(counts) / max(len(counts), 1))}
