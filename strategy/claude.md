# W4 recovery strategy — transform catalog

Working document. Goal: find a recipe that restores FP16-level **graded closed-loop safety**
on OpenDriveVLA-0.5B at ~4 bits/weight, where group-wise, AWQ and DCT rotation have all failed.

## The triage question

Every entry below is tagged by what it actually changes:

- **[C] concentration** — redistributes quantization error so no channel/weight dominates a shared
  scale. *Three independent refutations already on this model* (group-wise, AWQ, DCT rotation all
  land at 32–43% stationary early-crash vs 0% for FP16/W8).
- **[E] error compensation** — reduces committed error by adjusting not-yet-quantized weights.
  Untested here.
- **[K] capacity** — changes how many distinct values the grid can represent per weight, or how
  bits are distributed. Untested here.
- **[S] structure** — changes the parameterization/shape rather than the grid.

Prior from the mechanism analysis: the residual failure looks capacity-limited, not
concentration-limited. **[K]** and **[E]** are where to spend effort; **[C]** entries are listed
for completeness and for composition, not as standalone bets.

---

## Family A — Real orthogonal transforms  [C]

Norm-preserving, exactly invertible, foldable into weights (`y = Wx = (WQᵀ)(Qx)`).

**Fixed / data-free**
- Hadamard — Sylvester construction (power-of-two only), Paley construction (p+1 for prime p),
  Williamson, Baumert–Hall. Qwen dims 896 and 4864 are not powers of two; 896 = 2⁷·7 admits a
  Paley/Sylvester composite, which is *not* currently implemented (the code falls back to random
  orthogonal).
- Walsh–Hadamard (sequency-ordered Hadamard), fast transform O(n log n).
- DCT-I, DCT-II, DCT-III, DCT-IV. **DCT-II is what's implemented.** DCT-IV is the one used in
  lapped transforms and has different energy compaction.
- DST-I…IV (discrete sine).
- DFT / FFT (complex), and the **Discrete Hartley Transform** — the real-valued analogue of DFT,
  never tried in this literature as far as I know.
- Haar wavelet; Daubechies-k; Coiflets; Symlets; biorthogonal wavelets.
- Slant transform, Walsh, Rademacher.
- Number Theoretic Transform (NTT) — FFT over a finite field, exact integer arithmetic.

**Structured / sparse-composable**
- Givens rotations (2×2 plane rotations; any orthogonal matrix = product of ≤ n(n−1)/2 Givens).
- Householder reflections (I − 2vvᵀ); any orthogonal = product of ≤ n Householders.
- Butterfly matrices / Monarch matrices — sub-quadratic structured orthogonal families.
- Block-diagonal orthogonal (rotate within blocks only — cheap, composes with group-wise scales).
- Signed permutation matrices (cheapest non-trivial orthogonal).
- Permutation matrices (pure reordering; DuQuant's zigzag permutation lives here).
- Kac random walk (random Givens chain approximating Haar measure).

**Data-dependent**
- Karhunen–Loève / PCA rotation — diagonalizes the covariance; optimal decorrelation.
  (SliceGPT uses this for rotation+slicing.)
- ZCA whitening (symmetric whitening, minimal distortion), PCA whitening, Cholesky whitening.
- Procrustes alignment (nearest orthogonal to a target).
- Learned rotations on the Stiefel manifold, via Cayley parameterization or matrix exponential of
  a skew-symmetric generator. **This is SpinQuant.** Not implemented here.

---

## Family B — Quantum-computing unitaries  [C], mostly

All unitary, so all are "rotations" in the incoherence-processing sense. The interesting ones are
those that give *Haar-like randomness cheaply* or give *structured control over the spectrum*.

**Single/two-qubit gates (building blocks)**
- Pauli X, Y, Z; Hadamard H; phase gates S, T; rotations Rx(θ), Ry(θ), Rz(θ); U3 general.
- CNOT, CZ, SWAP, iSWAP, √SWAP, Toffoli, Fredkin, controlled-U.

**Composite structures**
- Quantum Fourier Transform — literally the DFT, with an O(log² n) circuit factorization.
- Clifford group elements; random Clifford sampling.
- **Unitary t-designs / 2-designs** — distributions matching Haar moments up to order t, realizable
  with far fewer gates than a full Haar unitary. *Directly relevant:* incoherence processing wants
  "random-enough" rotation cheaply, and a 2-design is exactly the formalization of "random enough
  for second-moment purposes."
- Reck and Clements decompositions — any unitary as a mesh of two-level (Givens) rotations.
- Quantum Shannon decomposition (recursive CS decomposition).
- Matchgates / fermionic linear optics (efficiently simulable subgroup).
- Trotter–Suzuki product formulas (approximating exp(iHt) by a product of simple unitaries).
- Grover / amplitude-amplification operator (product of two reflections).

**Spectral-transform frameworks**
- **Quantum Singular Value Transformation (QSVT)** and qubitization — apply a polynomial function
  to the *singular values* of an embedded matrix. Conceptually the most interesting entry here:
  it's a principled framework for reshaping a spectrum rather than rotating a basis. Classically
  this is just polynomial filtering of singular values (see Family D), and that framing is the
  usable one.
- Linear Combination of Unitaries (LCU) — express a non-unitary operator as Σ cᵢUᵢ.
- Block-encoding — embed an arbitrary matrix in a larger unitary.

*Honest note:* quantum gates buy nothing quantum here — we run classically, so a "quantum" unitary
is just a structured orthogonal matrix. The transferable ideas are **t-designs** (cheap Haar-like
randomness) and **QSVT/polynomial spectral filtering** (reshaping singular values). Everything else
in this family reduces to Family A.

---

## Family C — Non-orthogonal invertible transforms  [C]

Not norm-preserving, so they can *equalize* rather than merely rotate — strictly more expressive
than Family A, and the folding identity still holds for any invertible T: `W x = (W T⁻¹)(T x)`.

- Diagonal scaling — per-input-channel. **This is AWQ (salience) and SmoothQuant (migration).**
  Implemented.
- General affine / GL(n) invertible transform. **This is AffineQuant.** Not implemented.
- Kronecker-factored affine transforms. **This is FlatQuant's family.** Not implemented.
- Shear / elementary / triangular (LU factor) transforms.
- Polar decomposition T = UP: separates an orthogonal part from a PSD stretch, letting you tune the
  stretch independently of the rotation.
- Circulant / Toeplitz / Hankel / Vandermonde / Cauchy — structured, fast-multiply.
- Low-displacement-rank matrices.
- Sherman–Morrison rank-one updates.

---

## Family D — Decompositions  [S], with [K] variants

- SVD; truncated/low-rank SVD; randomized SVD.
- **Polynomial filtering of singular values** — the classical reading of QSVT. Reshape the spectrum
  before quantizing.
- QR (Gram–Schmidt / Householder / Givens); RRQR.
- LU / LDU; **Cholesky and LDLᵀ** — *the numerical core of GPTQ's error compensation.*
- Eigendecomposition; Schur; Jordan; Hessenberg; Golub–Kahan bidiagonalization.
- CUR, interpolative decomposition, Nyström.
- NMF; sparse + low-rank (robust PCA).
- Tensor: CP/PARAFAC, Tucker, **Tensor-Train**, Tensor Ring, hierarchical Tucker, block-term.
- Kronecker-product factorization; K-FAC (Kronecker-factored curvature).
- Monarch and butterfly factorizations.
- Quantized weight + low-rank residual (W ≈ Q + AB). **[K]** — a genuinely capacity-adding hybrid;
  cheap, and not tried here.

---

## Family E — Grids and codebooks  [K] ← the untested family

Everything in the current stack is **scalar** quantization: one value, one grid point. The methods
that actually make 4-bit and sub-4-bit work in the LLM literature are not scalar.

- Uniform symmetric / asymmetric RTN. *Implemented.*
- **Clipping search** — MSE- or KL-optimal clip ratio instead of absmax. *Not implemented; ~20
  lines; usually the largest single RTN gain at 4 bits.*
- Non-uniform scalar codebooks — k-means / Lloyd-Max on the weight distribution.
  **This is SqueezeLLM** (sensitivity-weighted k-means).
- Logarithmic / power-of-two grids; NF4 (normal-float, information-theoretically matched to a
  Gaussian); AF4.
- **Vector quantization** — quantize d weights jointly. Strictly more capacity per bit than scalar
  at the same rate (sphere-packing gain). GPTVQ, VPTQ.
- **Lattice codebooks** — E8, Barnes–Wall, Leech, D4. Optimal sphere packings in their dimensions.
  **This is QuIP#** (E8 + Hadamard incoherence).
- **Additive / residual / multi-codebook quantization** — sum of several codebook entries.
  **This is AQLM.**
- **Trellis-coded quantization** — Viterbi over a state machine; approaches the rate-distortion
  bound. **This is QTIP.**
- Product quantization; residual VQ.
- Stochastic / dithered rounding; error-feedback rounding.
- **Adaptive rounding** — learn round-up-vs-down per weight rather than nearest.
  **This is AdaRound**, and the LDLQ rounding in QuIP.

---

## Family F — Error compensation  [E] ← also untested here

- **GPTQ** — sequential column-wise quantization with inverse-Hessian error propagation into the
  not-yet-quantized weights. Cholesky-based. *Absent from this stack entirely; also a
  reviewer-visible gap.*
- OBQ / Optimal Brain Quantization; OBS/OBD lineage.
- LDLQ (QuIP's rounding, LDLᵀ-based).
- AdaRound / BRECQ / QDrop — block-wise reconstruction with learned rounding.
- OmniQuant — learnable clipping + learnable equivalent transformation.
- Norm-Tweaking; Z-Fold; bias correction (match pre/post-quant activation means).
- Quantization-aware fine-tuning / LoRA-on-quantized (QLoRA-style recovery).

---

## Family G — Allocation  [K]

- Uniform bit-width. *Implemented — all 168 Linears identical.*
- Per-layer / per-module mixed precision (sensitivity-ranked).
- Per-channel mixed precision. **This is QVLA's contribution** (ICLR'26, manipulation).
- Bit allocation with pruning as 0-bit (also QVLA).
- Hessian-trace / Fisher-information-guided allocation (HAWQ lineage).
- Outcome-guided allocation — allocate by effect on a *task* metric. Our braking-onset signal
  lives here, and this is the one criterion nobody has used.
- Keep-sensitive-layers-in-higher-precision (DA-PTQ's structural-Jacobian variant).
- N:M structured sparsity (2:4) combined with quantization.

---

## Standing note

W8 scores 0% stationary early-crash and W4 ~35%, so **a feasible point provably exists between
them.** Any allocation scheme that can interpolate between 4 and 8 bits has a guaranteed solution;
the only open question is how low the average can go. That makes Family G the safest bet and
Family E the highest-upside one.

---

# The search plan

## Goal and workflow

Find a **4-bit** recipe over weights *and* activations whose closed-loop driving safety matches
W8. The loop, per candidate:

```
W, A  ──▶  Methods(W, A)  ──▶  W', A'  ──▶  run on driving benchmark  ──▶  score
```

**Success criterion** (must match W8, not merely beat naive W4):

| metric | W8 (target) | naive W4 (floor) |
|---|---|---|
| stationary early-crash rate | **0.0%**, CI [0.0, 0.0] | 32–43% |
| graded NCAP (16 instances × 10 seeds) | **4.50**, CI [3.82, 4.94] | 2.62 |
| free-drive route competence | **97.1%** | 18.6% |

A candidate is *certified* only if its paired difference from FP16 has a CI containing zero on
graded NCAP — the same bar W8 cleared.

## Compose exhaustively — but over the right space

The intent is full coverage of the composition space: method 1 + method 2 + method 3, method 2 +
method 1 + method 3, chains up to length 5, in every order. Two things have to be said about that
before it becomes a job script.

### 1. Raw permutation enumeration is not runnable

Ordered sequences of length 1–5 drawn from N methods number `Σₖ₌₁⁵ N!/(N−k)!`:

| N methods | sequences | at 8 min/screen | at 55 min/full |
|---|---|---|---|
| 6 | 1,236 | 6.9 days | 47 days |
| 10 | 36,100 | 200 days | 3.8 years |
| 20 | 1,984,000 | 30 years | 207 years |

The deadline is ~1 month out. Even six methods exhaustively certified is 47 days of continuous
closed-loop compute.

### 2. Most of those orderings are the same experiment

This is the load-bearing point, and it is algebra, not opinion:

- Two rotations compose to a rotation: `Q₂(Q₁x) = (Q₂Q₁)x`, and `Q₂Q₁ ∈ O(n)`.
- Two diagonal scalings compose to a diagonal scaling. Two permutations compose to a permutation.
- Therefore **any chain, in any order, of rotations, scalings and permutations is a single
  invertible matrix** `T ∈ GL(n)`, applied once as `Wx = (WT⁻¹)(Tx)`.

So permuting Families A, B and C against each other does not explore new function space — every
one of those millions of orderings lands somewhere inside "one general invertible transform,"
which is just Family C (AffineQuant/FlatQuant). Enumerating the orderings mostly re-runs the same
experiment under different names.

What genuinely does **not** collapse, because these are not linear maps composed on the operand:

- the **grid / codebook** (scalar vs k-means vs vector vs lattice vs trellis) — Family E
- the **rounding rule** (nearest vs adaptive vs LDLQ vs stochastic)
- **error compensation** (GPTQ is sequential and data-dependent; it does not commute with anything)
- **bit allocation** across layers/channels — Family G
- the **activation-side** configuration (its own bits, granularity, and whether it shares T)

### 3. The space we actually enumerate

A pipeline of independent slots, taken as a product rather than a sequence. Order *within* the
transform slot is meaningless (§2); order *between* slots is fixed by what operates on what.

| slot | options |
|---|---|
| 1. transform `T` | identity · diagonal (AWQ, SmoothQuant α-sweep) · rotation (DCT-II/IV, Hartley, Haar, Paley-Hadamard at 896, random-orth, 2-design) · permutation · rotation∘diagonal · learned affine · Kronecker-factored |
| 2. clipping | absmax · MSE-optimal search · KL-optimal · percentile |
| 3. grid | uniform sym · uniform asym · NF4 · k-means · vector (d=2,4) · E8 lattice · trellis |
| 4. rounding | nearest · stochastic · AdaRound · LDLQ |
| 5. compensation | none · GPTQ · bias correction |
| 6. allocation | uniform W4 · per-layer mixed · per-channel mixed · onset-guided |
| 7. activation | bits ∈ {4,8,16} · per-token/per-tensor · shares `T` or not |

Full cross-product is ~20k cells; that is still too many to certify, so it is walked by the
staged protocol below rather than enumerated flat. Chains of length up to 5 are expressed here as
picking a non-identity option in up to 5 slots — which is the same coverage, minus the redundancy.

## Staged protocol

Measured costs on this harness: full closed-loop is **55 min** (16 instances × 10 seeds); the
stationary subset at 3 seeds is **~8 min** and carries essentially all of the signal, since every
4-bit failure to date is concentrated there.

- **S0 — replay screen (seconds).** Decoder-only replay of the 162 captured inputs. Rejects
  catastrophic candidates only (all-zero plan count, |xy| > 50 m, L2 blow-up). *Cannot certify —
  open-loop L2 is blind to exactly the damage we care about. Use it to kill, never to promote.*
- **S1 — stationary screen (~8 min).** 8 stationary scenarios × 3 seeds. Primary metric:
  early-crash rate. Promote anything with early-crash < 10%.
- **S2 — full adversarial (~55 min).** 16 instances × 10 seeds, graded NCAP with bootstrap CIs.
- **S3 — certification.** Add free-drive route competence to rule out degenerate freeze solutions,
  and confirm on held-out scenarios not used in any earlier stage.

**Held-out discipline:** slots 2–6 are tuned only on scenarios in the S1 development set.
Certification in S3 must use scenarios never used to select a candidate, or the result is
circular — fatal for a paper whose thesis is that metrics mislead.

## Ordering of the sweep

Search the axes the evidence favours first, so the budget is not spent re-refuting Family A:

1. **Grid and rounding** (slots 3–4) at fixed identity transform — the untested capacity axis.
2. **Error compensation** (slot 5) — GPTQ, absent from the stack and cheap.
3. **Allocation** (slot 6) — guaranteed feasible point exists between W4 and W8.
4. **Transform** (slot 1) as *preprocessing for whatever wins above*, not as a standalone bet.
   QuIP# is the template: the Hadamard rotation is not the win, it is what makes the lattice
   codebook work.
5. **Activations** (slot 7) last, since W4A4+DCT already sits at 32.5% and activation quant is not
   the binding constraint on the weight-side failure.
