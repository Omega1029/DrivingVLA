"""Rotation (incoherence-processing) + activation fake-quant for the OpenDriveVLA planner.

Ported from the OpenVLA-OFT CoRL study (offline-robotcs/quant_bench/quant_expirements/
{rotations,quant,quant_advanced}.py) for the ICRA cross-embodiment experiments (E1):
does the manipulation-side fix (orthogonal rotation) transfer to a text-head driving VLA?

    y = Wx = (W Q^T)(Q x)   for orthogonal Q  ->  quantize the *rotated* operands.

Qwen2.5-0.5B dims (hidden 896, intermediate 4864) are not powers of two, so exact
Sylvester-Hadamard does not exist; DCT-II (orthonormal, any n) is the principled pick --
the same conclusion the CoRL study reached for Llama's 11008-dim down_proj.

Usage (module-wrapping; composable with the existing QuantState weight-only path):
    from inference.rotquant import apply_rotated, remove_rotated
    apply_rotated(model, w_bits=4, a_bits=4, kind="dct", group=0)   # wrap backbone linears
    remove_rotated(model)                                           # restore originals
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.quantizers import _quantize_weight  # per-channel/group symmetric RTN


# ── orthogonal transforms ────────────────────────────────────────────────────
def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II, valid for any n (incl. non-pow2 like 896/4864)."""
    k = torch.arange(n).float().view(-1, 1)
    j = torch.arange(n).float().view(1, -1)
    M = torch.cos(math.pi / n * (j + 0.5) * k) * math.sqrt(2.0 / n)
    M[0] *= 1 / math.sqrt(2)
    return M


def hadamard_matrix(n: int) -> torch.Tensor:
    assert _is_pow2(n), f"hadamard needs power-of-two dim, got {n}"
    H = torch.ones(1, 1)
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    for _ in range(int(round(math.log2(n)))):
        H = torch.kron(H, base)
    return H / math.sqrt(n)


def random_orthogonal(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed + n)
    A = torch.randn(n, n, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    Q *= torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q


def make_rotation(n: int, kind: str) -> torch.Tensor:
    """'hadamard' falls back to random-orth for non-pow2 dims (mirrors CoRL code)."""
    if kind == "dct":
        return dct_matrix(n)
    if kind == "hadamard" and _is_pow2(n):
        return hadamard_matrix(n)
    if kind in ("hadamard", "random_orthogonal", "orth"):
        return random_orthogonal(n)
    if kind in ("none", "identity"):
        return torch.eye(n)
    raise ValueError(f"unknown rotation kind: {kind}")


# ── activation fake-quant ────────────────────────────────────────────────────
def quant_act(x: torch.Tensor, bits: int, mode: str = "per_token") -> torch.Tensor:
    """Symmetric dynamic activation quant (per_token = one scale per seq position)."""
    if bits is None or bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    if mode == "per_token":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    else:  # per_tensor
        scale = x.abs().max().clamp_min(1e-8) / qmax
    return torch.round((x / scale).clamp(-qmax - 1, qmax)) * scale


# ── the wrapped Linear ───────────────────────────────────────────────────────
class RotatedQuantLinear(nn.Module):
    """y = quant_act(x Q^T) @ quant_w(W Q^T)^T + b. Exact when Q orthogonal & no quant.
    Supports group-wise weight quant post-rotation (rotation x granularity crossover).
    kind='none' (Q=I) gives pure activation-quant / un-rotated W-A baselines."""

    def __init__(self, linear: nn.Linear, Q: torch.Tensor, w_bits: int,
                 a_bits: int = 0, group: int = 0, a_mode: str = "per_token",
                 Q_buffer: torch.Tensor | None = None, identity: bool = False):
        super().__init__()
        dtype = linear.weight.dtype
        self.identity = identity
        W = linear.weight.data.float()
        Wr = W if identity else W @ Q.float().t()
        if w_bits and 0 < w_bits < 16:
            Wr = _quantize_weight(Wr, w_bits, group, asym=False)
        self.register_buffer("weight", Wr.to(dtype))
        if not identity:
            if Q_buffer is None:
                Q_buffer = Q.to(dtype)
            self.register_buffer("Q", Q_buffer)
        self.bias = linear.bias
        self.a_bits = a_bits
        self.a_mode = a_mode
        self._orig = linear  # kept for exact restore

    def forward(self, x):
        if self.identity:
            xr = x
        else:
            xr = (x.to(self.Q.dtype) @ self.Q.t())
        if self.a_bits and self.a_bits < 16:
            xr = quant_act(xr.float(), self.a_bits, self.a_mode).to(self.weight.dtype)
        else:
            xr = xr.to(self.weight.dtype)
        return F.linear(xr, self.weight, self.bias)


# ── apply / remove over the Qwen backbone ────────────────────────────────────
def _backbone_linear_items(model):
    """Yield (parent, attr, linear) for the Qwen decoder-stack Linears (mirrors
    quantizers._iter_backbone_linears but with parent handles for module swap)."""
    backbone = model
    for _ in range(4):
        if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
            backbone = backbone.model
            break
        backbone = getattr(backbone, "model", backbone)
    for layer in backbone.layers:
        for holder_name in ("self_attn", "mlp"):
            holder = getattr(layer, holder_name, None)
            if holder is None:
                continue
            for attr, mod in vars(holder)["_modules"].items():
                if isinstance(mod, nn.Linear):
                    yield holder, attr, mod


def apply_rotated(model, w_bits: int = 4, a_bits: int = 4, kind: str = "dct",
                  group: int = 0, a_mode: str = "per_token") -> dict:
    """Wrap every backbone Linear. kind='none' => no rotation (A-quant baseline)."""
    remove_rotated(model)  # idempotent
    dev = next(model.parameters()).device
    identity = kind in ("none", "identity")
    rotations: dict[int, torch.Tensor] = {}
    buffers: dict[int, torch.Tensor] = {}
    n = 0
    items = list(_backbone_linear_items(model))
    mdtype = items[0][2].weight.dtype if items else torch.float16
    for parent, attr, lin in items:
        d = lin.in_features
        if not identity and d not in rotations:
            rotations[d] = make_rotation(d, kind).to(dev)
            buffers[d] = rotations[d].to(mdtype)
        Q = rotations.get(d, torch.empty(0))
        setattr(parent, attr, RotatedQuantLinear(
            lin, Q, w_bits, a_bits=a_bits, group=group, a_mode=a_mode,
            Q_buffer=buffers.get(d), identity=identity).to(dev))
        n += 1
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    cfg = {"mode": f"rot_{kind}", "w_bits": w_bits, "a_bits": a_bits,
           "group": group, "n_wrapped": n, "dims": sorted(rotations.keys())}
    print(f">>> apply_rotated: {cfg}")
    return cfg


def remove_rotated(model) -> int:
    """Restore original nn.Linear modules (exact: originals were kept by reference)."""
    n = 0
    for parent, attr, mod in list(_backbone_linear_items_any(model)):
        if isinstance(mod, RotatedQuantLinear):
            setattr(parent, attr, mod._orig)
            n += 1
    if n and next(model.parameters()).device.type == "cuda":
        torch.cuda.empty_cache()
    return n


def _backbone_linear_items_any(model):
    """Like _backbone_linear_items but yields wrapped modules too (for removal)."""
    backbone = model
    for _ in range(4):
        if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
            backbone = backbone.model
            break
        backbone = getattr(backbone, "model", backbone)
    for layer in backbone.layers:
        for holder_name in ("self_attn", "mlp"):
            holder = getattr(layer, holder_name, None)
            if holder is None:
                continue
            for attr, mod in list(vars(holder)["_modules"].items()):
                if isinstance(mod, (nn.Linear, RotatedQuantLinear)):
                    yield holder, attr, mod
