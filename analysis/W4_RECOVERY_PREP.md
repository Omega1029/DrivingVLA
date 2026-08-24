# W4 recovery: what's already built, what's missing, what the evidence implies

Prep audit ahead of trying new weight transforms. Written 2026-08-24.

## Plug-in surface

| where | what it does |
|---|---|
| `inference/quantizers.py::_quantize_weight` | the grid: symmetric/asymmetric RTN, per-out-channel (`group=0`) or group-wise. **Accepts any bit-width** — W5/W6/W7 need no new code. |
| `inference/quantizers.py::QuantState.apply(bits, group, mode)` | restores the FP snapshot then re-quantizes in place. Uniform bits across all 168 backbone Linears. |
| `inference/quantizers.py::QuantState.calibrate_awq` | forward-hook calibration, `s = mean(|x|)^alpha` normalized to unit mean. |
| `inference/rotquant.py::make_rotation` | `dct` (any n), `hadamard` (pow-2 only), `random_orthogonal` (QR). |
| `inference/rotquant.py::RotatedQuantLinear` | `y = quant_act(x Qᵀ) @ quant_w(W Qᵀ)ᵀ`; composes rotation × group × activation bits. |
| `inference/server.py::/quantize` | `{bits, group, mode, a_bits, rot}` — live re-quant, no model reload. A new mode needs a string here and a branch in `apply`. |

Only the Qwen decoder-stack Linears are touched. Perception, projectors, embeddings and
`lm_head` stay FP, so any result is attributable to the language/action head.

## Already tried, all landing in the same place

Stationary-scenario early-crash rate (`analysis/mechanism_all_configs.py`):
FP16 **0.0%**, W8 **0.0%**, group-W4 42.9%, AWQ-W4 37.5%, W4A4+DCT 32.5%.

## The caution this implies

Rotations, per-channel scalings and permutations are **reparameterizations**. They redistribute
quantization error; they do not add representational capacity. They pay off when error is
*concentrated* — a few outlier weights or channels inflating a shared scale.

Our own data says concentration is already being handled and it isn't enough:

- **group-wise scales** confine an outlier's damage to its own block — fully restores generation
  coherence (73.5% vs FP16's 76.0%) — and leaves safety at naive-W4 level.
- **AWQ** protects high-activation channels by salience. Same failure, 37.5%.
- **DCT rotation** makes operands near-Gaussian so no channel dominates. Same failure, 32.5%.

Three mechanically different anti-concentration methods, one outcome. That is meaningful evidence
the residual 4-bit failure is **capacity-limited, not concentration-limited** — in which case a
fourth transform of the same family should be expected to land at ~35% too.

## What is different *in kind* (and cheap)

1. **Clipping search.** Scales are currently `absmax` (`_quantize_weight` lines 53-61), the most
   outlier-sensitive possible choice. An MSE-optimal clip-ratio grid search is ~20 lines and is
   usually the single largest RTN improvement at 4 bits. Nothing else here does it.
2. **GPTQ.** Not implemented anywhere. Unlike everything above it *compensates* error — it updates
   the not-yet-quantized weights to cancel the error already committed, via the inverse Hessian of
   the layer input. Different objective, not a different grid. This is the standard 4-bit baseline
   and its absence is also a reviewer-visible gap.
3. **Non-uniform bit allocation.** `apply()` is uniform over all 168 Linears. W8 scores 0% early
   crash and W4 scores ~35%, so **a feasible point provably exists between them** — the open
   question is only how low the average bit-width can go. Allocation guided by the decision-timing
   signal is the one direction with both a guaranteed solution and a novel criterion.

## Triage rule for incoming proposals

Ask of each: does it reduce error *concentration* (rotation, permutation, smoothing, salience
scaling), or does it *compensate* error / *reallocate capacity*? The first family has three
independent refutations on this model already. The second is untested here.
