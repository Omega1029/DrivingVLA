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

## 2026-08-28 — draft clean versions of the remaining 5 confirmed-buggy scripts — PASS

Cloud routine cycle. Picked the next unclaimed daily-queue item: "draft cleaned versions of any
other script confirmed buggy above" (the five from the 2026-08-27 10:05 audit table). Read-only
audit + drafting only, no GPU/renderer access from this sandbox — nothing was run.

Drafted five new scripts in `closed_loop_harness/` (originals untouched, still snapshotted
read-only in `external_reference/`):

- **`run_positive_hunt_clean.sh`** (from `run_positive_hunt.sh`): same per-scene `launch_lane`
  pattern as `run_freedrive.sh`. Fix mirrors `run_freedrive_clean.sh` exactly — capture
  `/get_actors` once right after each fresh launch, `/update_actors` restore before every
  main.py invocation (both the freedrive run and every adversarial-category run, for every
  config). Also swapped the original's fire-and-forget `pkill -9 -f` for the wait-for-actual-
  release `kill_port` helper used in `run_sweep_clean.sh`, since that's the same "kill without
  confirming release" defect documented in that script's own comments — a related bug in the
  same class this loop exists to catch, not a new one I went looking for.
- **`parallel_benchmark_clean.sh`** (from `parallel_benchmark.sh`): same per-scene renderer /
  looped-configs-and-categories pattern across 4 lanes. Same fix: capture pristine actors once
  per scene after the renderer comes up, restore before every (category, config) main.py call.
  Same `kill_port`-for-`pkill` swap as above, for the same reason.
- **`run_quant_sweep_clean.sh`** (from `run_quant_sweep.sh`) and **`run_quant_full_clean.sh`**
  (from `run_quant_full.sh`): these two don't launch a renderer at all — they assume one is
  already up on :8000 and loop configs against it. Fix: capture whatever actor state is present
  on :8000 once at script start, restore it before every rollout (including, for
  `run_quant_full_clean.sh`, the AWQ calibration rollout). Flagged explicitly in each script's
  header comment: this only guarantees identical state *across configs within one run*, not that
  the captured state is the scene's true pristine state, since the script never owns the launch —
  operators must run these immediately after a fresh render-server start on scene 0103.
- **`run_awq_clean.sh`** (from `run_awq.sh`): same no-launch, assumes-live-server shape. Restores
  actor state before the calibration rollout and both AWQ-W4 scoring rollouts. Header repeats the
  2026-08-27 10:05 second-order-contamination flag verbatim: this fix does not retroactively
  clean the *original* `logs/awq_scales.pt` if that was calibrated on a mutated renderer —
  recalibration from scratch with this version would be needed to trust it, and that's a human
  call, not something to decide here.

All five: `bash -n` syntax-checked clean. Not executed — no GPU/renderer access from this
sandbox, and even with access these are drafts pending human review before first use (per
AUTOLOOP.md's execution-modes split). No `.tex` file touched; no claim, number, or interpretation
changed anywhere. `analysis/verify_before_commit.sh` passed (no `.tex` changed, no closed-loop
output dir changed, no force-push/off-branch-push patterns in the new scripts).

Marked the queue item `[x]` in AUTOLOOP.md. Next unclaimed cloud-queue items: attempting to
compile the two papers with a user-local LaTeX toolchain, and the two static-review items on
`analysis/*.py` and `four_bits_without_loss.tex` vs. `RESULTS_clean_harness.md`.

## 2026-08-29 — compile papers/four_bits_without_loss.tex and papers/icra27_crossembodiment.tex — PASS

Cloud routine cycle. Picked the next unclaimed daily-queue item: attempt to compile the two
papers, retrying the LaTeX toolchain question from 2026-08-26 (neither `pdflatex` nor `tectonic`
was available then).

This sandbox has `sudo apt-get` access to Ubuntu's `noble` archive (a couple of unrelated PPAs
403'd, the main archive/security/updates mirrors did not), so installed a minimal toolchain
rather than fighting `cargo install tectonic` through the proxy allowlist:
`texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-publishers`
(`--no-install-recommends`). `texlive-publishers` was needed for the `IEEEtran` document class
both papers use; `amsmath`/`amssymb`/`booktabs`/`graphicx`/`xcolor`/`hyperref` (the only other
packages either file `\usepackage`s) are covered by latex-base/latex-extra.

Compiled each paper twice with `pdflatex -interaction=nonstopmode -halt-on-error` into a scratch
output dir (not the repo — build artifacts aren't being committed):

- `four_bits_without_loss.tex`: pass 1 exit 0, produced a 4-page PDF; only warnings were
  forward-reference "undefined on first pass" (expected before a second pass) and one cosmetic
  `OT1/ptm/m/scit` font-shape substitution warning (IEEEtran + Times small-caps-italic, harmless,
  falls back to plain italic). Pass 2 exit 0, zero undefined references, only the same font
  warning.
- `icra27_crossembodiment.tex`: pass 1 and pass 2 both exit 0, zero undefined references, same
  cosmetic font warning (three occurrences, one per use).

**No compile errors in either file** — no unbalanced environment, no broken `\ref`/`\label`, no
missing package. Since rule 1 only permits fixing actual compile errors and there were none, **no
`.tex` file was touched.** `git status` after the compile runs was clean (build output went to
the scratch dir, not the working tree).

Did not attempt `workshop1_openloop_blind.tex` or `workshop3_edge_memory.tex` — not in this
queue item's scope (both already have committed PDFs; the queue item names only the two papers
that don't: `four_bits_without_loss.tex` has no committed PDF, `icra27_crossembodiment.tex`'s
existing committed PDF was not diffed against this run's output since that's outside what "attempt
to compile" asked for).

Marked the queue item `[x]` in AUTOLOOP.md. `analysis/verify_before_commit.sh` passed (no `.tex`
staged, so checks 1 and 3 skip; no closed-loop output dir changed, so check 2 skips; no `.sh`
changes; PASSED). Only `AUTOLOOP.md` and this `PROGRESS.md` entry are staged for commit. Next
unclaimed cloud-queue items: the two static-review items on `analysis/*.py`'s bootstrap/outcome-
classification correctness, and `four_bits_without_loss.tex`'s claims cross-checked sentence by
sentence against `RESULTS_clean_harness.md`/`PROGRESS.md`/git log.
