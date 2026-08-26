"""Quantization grids and codebooks for the W4-recovery sweep.

Everything in the existing stack is uniform-RTN with an absmax scale. This module adds the
axes that are *capacity* changes rather than basis changes (see strategy/claude.md):

  clip search : MSE-optimal clip ratio instead of absmax  -- cheapest known RTN win at 4 bits
  nf4         : NormalFloat4, levels are standard-normal quantiles (information-matched)
  kmeans      : non-uniform scalar codebook, batched Lloyd per group (SqueezeLLM-style grid)
  vq          : vector quantization over d weights jointly -- sphere-packing gain over scalar

All operate on the last dim of a (..., n) tensor and return a dequantized tensor of the same
shape and dtype (fake quant), so they drop into the existing weight path unchanged.
"""
from __future__ import annotations

import torch

# QLoRA NF4 levels: quantiles of a standard normal, normalized to [-1, 1].
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])


def _rtn(w: torch.Tensor, bits: int, asym: bool, clip: float = 1.0) -> torch.Tensor:
    """Uniform RTN along the last dim, with an optional clip ratio on the range."""
    if asym:
        qmax = 2 ** bits - 1
        wmin = w.amin(dim=-1, keepdim=True) * clip
        wmax = w.amax(dim=-1, keepdim=True) * clip
        scale = (wmax - wmin).clamp_(min=1e-8) / qmax
        zp = torch.round(-wmin / scale)
        return (torch.clamp(torch.round(w / scale) + zp, 0, qmax) - zp) * scale
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    scale = (w.abs().amax(dim=-1, keepdim=True) * clip).clamp_(min=1e-8) / qmax
    return torch.clamp(torch.round(w / scale), qmin, qmax) * scale


def rtn_clipsearch(w: torch.Tensor, bits: int, asym: bool,
                   ratios=None, n_ratio: int = 20) -> torch.Tensor:
    """Pick, per group, the clip ratio minimising squared error. Absmax is ratio 1.0."""
    if ratios is None:
        ratios = torch.linspace(0.5, 1.0, n_ratio).tolist()
    best = None
    best_err = None
    for r in ratios:
        q = _rtn(w, bits, asym, clip=r)
        err = (q - w).pow(2).sum(dim=-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            take = err < best_err
            best = torch.where(take, q, best)
            best_err = torch.where(take, err, best_err)
    return best


def nf4(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """NormalFloat: scale each group by its absmax, snap to normal-quantile levels."""
    lv = NF4_LEVELS.to(w.device, w.dtype)
    if bits != 4:
        # generic normal-float: quantiles of N(0,1) at 2^bits points, normalised
        k = 2 ** bits
        p = torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=w.device, dtype=torch.float32)
        lv = torch.special.ndtri(p).to(w.dtype)
        lv = lv / lv.abs().max()
    s = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
    idx = torch.argmin((w.unsqueeze(-1) / s.unsqueeze(-1) - lv).abs(), dim=-1)
    return lv[idx] * s


def kmeans(w: torch.Tensor, bits: int, iters: int = 12) -> torch.Tensor:
    """Non-uniform scalar codebook, ONE codebook shared across the whole tensor.

    Bit accounting: log2(K)=bits per weight, plus a per-group scale (same cost as the
    existing group-wise path) plus 2^bits shared fp16 codebook entries per tensor, which is
    negligible. Fitting a codebook *per group* would smuggle in extra bits -- the codebook
    would be larger than the data it encodes -- so the scope here is deliberate.

    w: (G, n) -> (G, n).
    """
    K = 2 ** bits
    s = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
    wn = (w / s).reshape(-1)                                   # normalised, whole tensor
    lo, hi = wn.amin(), wn.amax()
    c = lo + (hi - lo) * torch.linspace(0, 1, K, device=w.device, dtype=w.dtype)
    for _ in range(iters):
        a = (wn.unsqueeze(1) - c.unsqueeze(0)).abs().argmin(dim=1)
        tot = torch.zeros(K, device=w.device, dtype=w.dtype).scatter_add_(0, a, wn)
        cnt = torch.zeros(K, device=w.device, dtype=w.dtype).scatter_add_(
            0, a, torch.ones_like(wn))
        c = torch.where(cnt > 0, tot / cnt.clamp(min=1), c)
    a = (wn.unsqueeze(1) - c.unsqueeze(0)).abs().argmin(dim=1)
    return (c[a].reshape(w.shape)) * s


def vq(w: torch.Tensor, bits: int, dim: int = 2, iters: int = 10) -> torch.Tensor:
    """Vector quantization over `dim` weights jointly, ONE codebook per tensor.

    K = 2^(bits*dim) entries shared tensor-wide, so the rate really is `bits` per weight.
    (A per-group codebook here would be degenerate: with group 128 and dim 2 there are only
    64 vectors per group against 256 codewords, i.e. lossless but not 4-bit.)

    w: (G, n), n % dim == 0.
    """
    G, n = w.shape
    assert n % dim == 0, f"group {n} not divisible by vq dim {dim}"
    K = 2 ** (bits * dim)
    s = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
    v = (w / s).reshape(-1, dim)                                # (N, dim) whole tensor
    N = v.shape[0]
    if K >= N:
        raise ValueError(f"vq codebook K={K} >= N={N} vectors: rate claim would be false")
    step = max(N // K, 1)
    c = v[torch.arange(K, device=w.device) * step % N].clone()  # (K, dim)
    for _ in range(iters):
        a = torch.cdist(v, c).argmin(dim=1)
        oh_tot = torch.zeros(K, dim, device=w.device, dtype=w.dtype).index_add_(0, a, v)
        cnt = torch.zeros(K, device=w.device, dtype=w.dtype).index_add_(
            0, a, torch.ones(N, device=w.device, dtype=w.dtype)).unsqueeze(1)
        c = torch.where(cnt > 0, oh_tot / cnt.clamp(min=1), c)
    a = torch.cdist(v, c).argmin(dim=1)
    return c[a].reshape(w.shape) * s


GRIDS = ("uniform", "uniform_asym", "clip", "clip_asym", "nf4", "kmeans", "vq2")


def quantize_grid(w2: torch.Tensor, bits: int, grid: str) -> torch.Tensor:
    """Dispatch. w2 is (G, n): quantization is along the last dim."""
    if grid == "uniform":
        return _rtn(w2, bits, asym=False)
    if grid == "uniform_asym":
        return _rtn(w2, bits, asym=True)
    if grid == "clip":
        return rtn_clipsearch(w2, bits, asym=False)
    if grid == "clip_asym":
        return rtn_clipsearch(w2, bits, asym=True)
    if grid == "nf4":
        return nf4(w2, bits)
    if grid == "kmeans":
        return kmeans(w2, bits)
    if grid == "vq2":
        return vq(w2, bits, dim=2)
    raise ValueError(f"unknown grid: {grid}")
