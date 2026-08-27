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
2. **Never `git push --force`, never push to `master`/`main`.** All work stays on
   `auto/opendrivevla-verify`. Merging to `mechanism-analysis` or `master` is a human decision.
3. **A commit requires the verification gate to pass** (`analysis/verify_before_commit.sh`). If
   it fails, fix the specific failure or flag it — do not commit around it.
4. **One task per cycle.** Pull the next unclaimed item from the queue below, do it, log it,
   stop. Don't chain into the next task in the same invocation — that's how scope creep and
   compounding errors happen unattended.
5. **If GPU state is ambiguous** (renderer/model processes already running, ports already bound,
   an in-progress sweep with unknown ownership) — check `PROGRESS.md` for the last entry before
   touching anything. If it's not clear whether a previous cycle's job finished, flag it and stop
   rather than guessing.
6. **Budget**: see the cycle-level token/dollar cap set at schedule time. If a task will exceed
   it, do the read-only / analysis half and flag the compute-heavy half for a human-approved cycle.

## Task queue (ordered — top item is next)

- [ ] **Audit `~/neuroncap/neuro-ncap/run_freedrive.sh` for the renderer-reuse bug.** Already
  confirmed structurally similar to the buggy `run_benchmark_12.sh` (see PROGRESS.md 2026-08-27):
  `launch_lane` once per scene, configs looped inside, no `update_actors` restore call in the
  file. Confirm by inspection, then decide: if confirmed, write a `run_freedrive_clean.sh`
  following the restore pattern in `closed_loop_harness/run_sweep_clean.sh` (do not run it yet —
  that's compute-heavy and belongs in the next cycle or a human-approved one).
- [ ] **Re-run free-drive under the restore protocol** (only after the above is written and
  reviewed). Compare against the currently-reported W8 97.1% / naive W4 18.6% table. Log the
  result in PROGRESS.md regardless of outcome. Do NOT touch any `.tex` file with the result.
- [ ] **Audit `run_positive_hunt.sh`, `run_awq.sh`, `run_quant_full.sh`, `run_quant_sweep.sh`**
  for the same pattern. These generated the AWQ/W4A4+DCT numbers currently in
  `analysis/mechanism_all_configs.py`'s inputs (already-void per the correction, but the raw
  sweep scripts haven't been individually confirmed clean or dirty — worth knowing for any future
  reuse of that data).
- [ ] **Attempt a LaTeX compile of `papers/four_bits_without_loss.tex` and
  `papers/icra27_crossembodiment.tex`.** No LaTeX toolchain was available as of 2026-08-26; check
  whether one can be installed without sudo (e.g. a user-local tectonic/texlive binary), and if
  so compile and report errors. Fix only compile errors (rule 1 still applies to content).
- [ ] (blocked on the above two) **If free-drive is confirmed contaminated and re-measured**,
  prepare — but do not apply — a suggested diff to `papers/four_bits_without_loss.tex` and any
  other paper still citing the old free-drive numbers, and log it in PROGRESS.md for human
  review.

## Verification gate

`analysis/verify_before_commit.sh` must exit 0 before any commit. It currently checks: LaTeX
environment balance on any changed `.tex` file, no dangling `\ref`/`\label` pairs, and (if a
closed-loop output directory changed) that `analysis/compare_clean_vs_archived.py` still runs
without error. Extend it as new checks become relevant — that's in-scope for this loop, since it
strengthens the gate rather than asserting a result.

## Every cycle ends with

An append to `PROGRESS.md` (never edit past entries), a commit on `auto/opendrivevla-verify` if
the gate passed, and a push. If nothing was accomplished (blocked, ambiguous state, budget), log
that too — a logged no-op is informative; a silent one looks like the loop died.
