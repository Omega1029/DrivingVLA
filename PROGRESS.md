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

## 2026-08-27 09:22 — config decision — N/A

User set cadence: once/day. And: no push for now -- scheduled cycles commit locally on
auto/opendrivevla-verify only; a human pushes when ready. AUTOLOOP.md rules 2 and the
end-of-cycle checklist updated accordingly.

## 2026-08-27 09:45 — config decision — N/A

Resolved via /schedule: cloud routines run in isolated per-firing sandboxes with no access to
this machine (no GPUs, no ~/neuroncap/, nothing outside the git checkout) and nothing persists
between firings except what's pushed. Consequences:
- GPU re-measurement can only ever be interactive/local, never part of the daily cloud loop.
  Moved to its own section in AUTOLOOP.md, explicitly marked do-not-attempt-from-cloud.
- Push is now allowed for the cloud routine specifically, scoped: `git push origin
  auto/opendrivevla-verify` only, never --force, never master/main, never another branch. Without
  this the daily loop is architecturally incapable of progress -- every firing would re-clone the
  same static state and reach the same conclusion forever.
- Copied 6 local-only harness scripts the cloud routine needs to audit into
  `closed_loop_harness/external_reference/` (snapshot, not a live mirror, per its README) since
  the cloud sandbox cannot see `~/neuroncap/neuro-ncap/` at all.
- AUTOLOOP.md's task queue split into a cloud-scoped daily section and an interactive-only
  section. Rules 2 and 5 rewritten accordingly.

Proceeding to create the RemoteTrigger routine: claude-sonnet-5, once/day, repo
github.com/Omega1029/DrivingVLA, branch auto/opendrivevla-verify.

## 2026-08-27 10:05 — audit: all remaining closed-loop scripts share the renderer-reuse defect — FLAGGED

Completed the cloud-queue audit item interactively (GPUs were idle and the free-drive re-run was
starting anyway). Checked all five remaining scripts in `closed_loop_harness/external_reference/`.
**All five have the defect.** Combined with the already-confirmed `run_benchmark_12.sh` and
`run_freedrive.sh`, that means *every* closed-loop generation script in this project except the
two written this week (`run_sweep_clean.sh`, `run_freedrive_clean.sh`) is contaminated.

| script | renderer launched | configs looped inside | `/update_actors` restores | verdict |
|---|---|---|---|---|
| `run_positive_hunt.sh` | per-scene (`launch_lane "$s"`, L48) | yes (L49) | **0** | CONFIRMED BUGGY |
| `parallel_benchmark.sh` | per-scene (L33-35) | yes (L41) | **0** | CONFIRMED BUGGY |
| `run_quant_sweep.sh` | none — assumes live server on :8000 | yes (L21) | **0** | CONFIRMED BUGGY |
| `run_quant_full.sh` | none — assumes live server on :8000 | yes | **0** | CONFIRMED BUGGY |
| `run_awq.sh` | none — assumes live server on :8000 | sequential rollouts | **0** | CONFIRMED BUGGY |

Two consequences worth escalating:

1. **`run_positive_hunt.sh` generated the W8 / AWQ-W4 / W4A4+DCT numbers**, and its config order is
   `w8 awq_w4_g128 w4a4_dct` — so W8 ran first on a clean renderer and both 4-bit variants ran on
   mutated ones. Same structure as the adversarial benchmark's fp16-first ordering. This is likely
   why W8 looked clean and the 4-bit recipes looked broken in that sweep too.
2. **`run_awq.sh` calibrates the AWQ scales during a live rollout** on a renderer that may already
   be mutated. If so, `logs/awq_scales.pt` was itself calibrated on contaminated activations — a
   second-order contamination affecting any result that used AWQ mode, not just the scores. Flagged
   for human review; not investigated further per AUTOLOOP rule (surprising findings get flagged,
   not chased alone).

No `.tex` file touched. Free-drive re-measurement under the restore protocol is running now
(`run_freedrive_clean.sh`, 14 scenes x 4 configs x 10 seeds, GPUs 1-4); result will be a separate
entry.
