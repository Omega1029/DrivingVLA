# Autonomous verification loop — operating rules

This file is read by every scheduled invocation. It is the only context those invocations start
with, besides `PROGRESS.md` and the repo itself — each cycle is a fresh, memoryless session, so
anything not written down here or in PROGRESS.md does not exist to it.

## Mission (read this first)

This project already had two "findings" turn out to be simulator artifacts that survived ten
seeds, paired comparisons, and bootstrap confidence intervals — see `PROGRESS.md`'s 2026-08-27
entry and the git history on `mechanism-analysis`. That happened because the harness was trusted
and the statistics were treated as sufficient evidence on their own.

**The job of this loop is verification and hardening, not discovery.** It audits existing
scripts for the specific class of bug already found (stateful services + fixed enumeration order
+ no restore/reset between configs), re-runs measurements under a corrected protocol, and reports
what it finds. It does not go looking for new positive results, and it does not iterate on a
measurement hoping for a better number. If a result is surprising, the correct action is to stop
and flag it in PROGRESS.md, not to investigate further alone or to keep re-running until it looks
right — surprising-and-then-explained-by-a-real-mechanism is fine; surprising-and-quietly-massaged
is exactly the failure this loop exists to prevent.

## Hard rules

1. **Never edit a `papers/*.tex` file's claims, numbers, or abstract based on a new finding.**
   Mechanical LaTeX fixes (a missing brace, a broken `\ref`, a compile error) are fine. A finding
   that would change what a paper asserts goes in PROGRESS.md as a flagged item for human review,
   full stop — even if you are confident it's correct. The last two times a result this project
   trusted got quietly wrong, it was because no one paused to ask why the harness behaved oddly
   until pushed.
2. **Push is allowed, narrowly.** `git push origin auto/opendrivevla-verify` only. Never
   `--force`, never push to `master`/`main`, never create or push any other branch. This is how
   a cloud routine's work survives past its own ephemeral sandbox — see "Execution modes" below
   for why that matters. If a push is rejected (non-fast-forward), pull/rebase onto the current
   branch tip first; if that doesn't resolve cleanly, stop and flag rather than force.
3. **A commit requires the verification gate to pass** (`analysis/verify_before_commit.sh`). If
   it fails, fix the specific failure or flag it — do not commit around it.
4. **One task per cycle.** Pull the next unclaimed item from the queue below, do it, log it,
   stop. Don't chain into the next task in the same invocation — that's how scope creep and
   compounding errors happen unattended.
5. **GPU work is manual-only, never part of the daily cloud loop.** The daily routine runs in
   an isolated cloud sandbox with no access to this machine's GPUs, the local NeuroNCAP/NeuRAD
   install, or the nuScenes data — it is architecturally incapable of re-measurement, and should
   never attempt it. Free-drive re-runs and any other GPU-dependent task in the queue are done
   interactively, by explicit request, in a local session. If that ever happens: before claiming
   GPUs, check `nvidia-smi --query-compute-apps` for unexpected existing usage and check recent
   `PROGRESS.md` entries for an in-progress sweep already launched; stop and flag rather than
   kill anything not traceable to a logged cycle of this project's own.
6. **Budget**: see the cycle-level token/dollar cap set at schedule time. If a task will exceed
   it, do the read-only / analysis half and flag the compute-heavy half for a human-approved cycle.

## Execution modes

This loop runs in two very different contexts and the task queue below is split accordingly.

**Daily cloud routine** (scheduled via claude.ai routines, `claude-sonnet-5`, once/day). Each
firing is a **fresh, isolated sandbox with its own git clone** — nothing on it persists between
firings except what gets pushed. It can read/write anything in this repo, including the snapshot
copies under `closed_loop_harness/external_reference/`, and it can install tooling (e.g. attempt
a LaTeX toolchain). It has **no access to GPUs, the local NeuroNCAP/NeuRAD install, the nuScenes
data, or anything outside this git checkout.** Because nothing survives without a push, this mode
must push (rule 2) — a no-push cloud cycle is indistinguishable from doing nothing, since the
next day's sandbox starts from the same state regardless of what the previous one did internally.

**Interactive/local session** (this machine, run by explicit request, not scheduled). Has real
access to GPUs 1-4, the local harness, and the actual `~/neuroncap/` install. This is where any
GPU-dependent task in the queue below actually executes — the daily cloud routine only prepares
for it (drafting scripts, confirming bug patterns by static audit) and flags it as ready.

## Task queue

Mark items `[x]` (with the commit hash) when done, in both this file and a PROGRESS.md entry.
Pull the next unclaimed `[ ]` item matching your execution mode; if none match, log a no-op.

### Daily cloud routine queue (git-scoped, no GPU needed)

- [x] Confirm `run_freedrive.sh` has the renderer-reuse bug by inspection. **Done 2026-08-27**
  (interactively, before the cloud routine existed): `launch_lane` once per scene, configs
  looped inside, no `update_actors` restore call anywhere in the file. Snapshot copied to
  `closed_loop_harness/external_reference/run_freedrive.sh`.
- [x] Audit the other five scripts in `closed_loop_harness/external_reference/`. **Done 2026-08-27**
  (interactively). All five CONFIRMED BUGGY — see PROGRESS.md 2026-08-27 10:05 for the table.
  Every closed-loop generation script in the project except `run_sweep_clean.sh` and
  `run_freedrive_clean.sh` has the defect. Two escalations flagged there: `run_positive_hunt.sh`
  ran W8 first (explaining why W8 looked clean and 4-bit didn't in that sweep), and `run_awq.sh`
  may have calibrated `logs/awq_scales.pt` itself on contaminated activations.
- [x] **Draft** `run_freedrive_clean.sh`. **Done 2026-08-27** (interactively — and run; see the
  interactive queue). Mirrors the restore protocol validated in `run_sweep_clean.sh`: pristine
  actor capture once per scene, `update_actors` restore before every invocation, PID-tracked
  teardown that waits for actual port release, staggered 4-lane start.
- [ ] Similarly draft cleaned versions of any other script confirmed buggy above.
- [ ] Attempt to compile `papers/four_bits_without_loss.tex` and `papers/icra27_crossembodiment.tex`.
  Try a user-local LaTeX toolchain (e.g. `tectonic` via cargo, or check for a preinstalled
  `pdflatex`/`tectonic`) since neither was available as of 2026-08-26. Fix only compile errors —
  rule 1 still applies to content.
- [ ] Static review of `analysis/*.py`: correctness of the bootstrap CI math, the outcome
  classification logic (`brake_onset_ci.py`'s early/frozen/moving split — see the survivorship
  bug it already caught once, documented in its own comments), and whether
  `RESULTS_clean_harness.md` accurately reflects what the scripts actually compute. This is
  read-only review — do not re-run anything GPU/renderer-dependent.
- [ ] Cross-check `papers/four_bits_without_loss.tex`'s claims against what's actually in
  `analysis/RESULTS_clean_harness.md`, `PROGRESS.md`, and the git log, sentence by sentence.
  Flag (do not fix) any claim that isn't traceable to committed evidence.

### Interactive/local-only queue (needs GPUs — do not attempt from the cloud routine)

- [ ] Once `run_freedrive_clean.sh` exists and has been reviewed: run it under the restore
  protocol, claiming GPUs 1-4. Compare against the currently-reported free-drive numbers
  (W8 97.1%, naive W4 18.6%). Log the result in PROGRESS.md regardless of outcome. Do NOT touch
  any `.tex` file with the result — flag it for review per rule 1.
- [ ] Extend the closed-loop CI tightening (more scenes/seeds) if there's compute budget and a
  human has asked for it specifically — not a default background task.

## Verification gate

`analysis/verify_before_commit.sh` must exit 0 before any commit. It currently checks: LaTeX
environment balance on any changed `.tex` file, no dangling `\ref`/`\label` pairs, and (if a
closed-loop output directory changed) that `analysis/compare_clean_vs_archived.py` still runs
without error. Extend it as new checks become relevant — that's in-scope for this loop, since it
strengthens the gate rather than asserting a result.

## Every cycle ends with

An append to `PROGRESS.md` (never edit past entries), marking the completed queue item `[x]`
with the commit hash, a commit on `auto/opendrivevla-verify` if the gate passed, and (cloud
routine only — see rule 2) a push. If nothing was accomplished (blocked, ambiguous state,
budget, no matching queue item for this execution mode), log that too — a logged no-op is
informative; a silent one looks like the loop died. Since a cloud cycle's sandbox is destroyed
after it ends, its final message must also state in full what it did and why, not just point at
a commit — that message is the only record if the push somehow fails.
