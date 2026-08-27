# OpenDriveVLA — autonomous verification log

One entry per cycle, newest first. Every entry is appended by a script or an autoloop cycle,
never rewritten — this file is the audit trail, not a status board.

Format: `## YYYY-MM-DD HH:MM — <task> — <PASS|FAIL|FLAGGED>` then a short body.

---

## 2026-08-27 — scaffolding created — N/A

Seeded this log and `AUTOLOOP.md` at the start of setting up scheduled autonomous verification
cycles. State as of this commit:

**Confirmed correct** (re-measured under the state-restore protocol, `closed_loop_harness/run_sweep_clean.sh`):
- Graded closed-loop safety (16 instances x 10 seeds): FP16 4.81 [4.62,4.97], W8 4.81 [4.58,5.00],
  naive W4 4.71 [4.38,5.00], group W4 g128 4.86 [4.67,4.98]. All indistinguishable from FP16.
- Activation-quantization crossover (renderer-independent replay, `icra27/e1/`): W16A4 alone
  inflates L2 47x; DCT rotation restores it. Never touched the renderer, unaffected by the bug.

**Known contaminated, corrected**: the original closed-loop safety table, coherence table, and
the braking-onset mechanism analysis (all in `output/benchmark12`, produced by
`closed_loop_harness/run_benchmark_12.sh`, which reuses one renderer per scene across configs
with no state restore).

**Suspected contaminated, NOT yet re-measured**: `~/neuroncap/neuro-ncap/run_freedrive.sh` has
the identical per-scene-then-loop-configs pattern (`launch_lane` once per scene, configs looped
inside, no `update_actors` restore call anywhere in the file). The free-drive route-competence
table (W8 97.1%, naive W4 18.6%, currently reported in prior papers) was produced by this script
and should be treated as unverified until re-measured under restore. Also unaudited:
`run_positive_hunt.sh`, `run_awq.sh`, `run_quant_full.sh`, `run_quant_sweep.sh`.

**Not evaluated at all**: the OpenVLA-OFT / LIBERO manipulation pipeline. Explicitly out of
scope for `papers/four_bits_without_loss.tex` per 2026-08-26 decision to drop that claim rather
than audit it under deadline pressure.

## 2026-08-27 09:15 — config decision — N/A

User authorized: re-measurement (free-drive re-run, and any future closed-loop re-measurement
task in the queue) may claim GPUs 1-4 itself without waiting for a human-kicked-off cycle. Added
the corresponding safety rule to AUTOLOOP.md (rule 5): check `nvidia-smi` and recent PROGRESS.md
entries before claiming; never kill a process this loop didn't start; ambiguous state still means
stop-and-flag, not proceed. Cadence and push-permission scope still open before scheduling starts.
