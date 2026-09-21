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

## 2026-08-30 13:19 — static review: analysis/*.py bootstrap/outcome-classification, RESULTS_clean_harness.md accuracy — FLAGGED

Cloud routine cycle. Picked the next unclaimed daily-queue item: static, read-only review of
`analysis/*.py` (bootstrap CI math, the early/frozen/moving outcome-classification logic, and
whether `RESULTS_clean_harness.md` matches what `compare_clean_vs_archived.py` actually computes).
No GPU/renderer access used or needed; no script was executed against real data (none is
reachable from this sandbox) — this was pure code reading and cross-referencing.

**Bootstrap CI math — no bug found.** `boot_ci`/`boot` (`brake_onset_ci.py`, `mechanism_all_configs.py`,
`compare_clean_vs_archived.py`) all do a standard percentile bootstrap: resample-with-replacement
means, `N_BOOT=10_000` times, take the values at `int(.025*n)`/`int(.975*n)` after sorting. That's
a correct percentile bootstrap (the index vs. true-percentile offset is a fraction of a percent at
n=10,000, not a bug). `brake_onset_ci.py::spearman` computes rank-based Pearson correlation with
proper average-rank tie handling, which is a correct Spearman's rho implementation.

**Outcome-classification logic — confirmed gap, same class as the bug the loop already fixed
once.** `brake_onset_ci.py::reach_profile` (L33-42) and `mechanism_all_configs.py::load_run`
(L36-51) build `p` by keeping only frames with `len(w) >= 6` waypoints, then check
`len(p) < MIN_FRAMES` to classify a short rollout as `"early"` (the fix the file's own comment
says was already made once, to stop excluding near-immediate crashes from the denominator as if
they were missing data). But **that check only runs if `p` is non-empty** — `if not p: return
None, False` (resp. `return None`) fires first when *every single frame* in the rollout is
malformed. `main()`/`collect()` treat that `None` as "no run" and skip it entirely (`if p is None:
continue`), so a rollout that produced zero valid frames — arguably the single worst outcome, a
total generation collapse from frame one — is silently dropped from every denominator instead of
being counted `"early"`. This is the exact survivorship pattern already caught and fixed for the
1-4-valid-frame case, just not extended to the 0-valid-frame case. Because a dropped run can only
ever be the worst-outcome case here (nothing else produces an empty profile), this can only bias a
config's reported early-crash rate **downward** (looks better than it is), never upward. I cannot
tell whether any real run actually hits zero valid frames — no run data is reachable from this
sandbox — so I don't know whether this changes any committed number, only that the code has the
gap. It's relevant because `analysis/W4_RECOVERY_PREP.md` cites exactly these early-crash-rate
numbers from `mechanism_all_configs.py` (FP16 0.0%, W8 0.0%, group-W4 42.9%, AWQ-W4 37.5%,
W4A4+DCT 32.5%) as load-bearing evidence for a strategic conclusion ("capacity-limited, not
concentration-limited") — worth re-checking once the underlying run data can be inspected.
Separately, `brake_diag.py::run_stats` (L28-29, `if len(reaches) < 5: return None`) still has the
*pre-fix* version of this same bug in a broader form: it drops any run with 1-4 valid frames
entirely (not just 0), rather than counting it as catastrophic the way `brake_onset_ci.py` now
does. I found no committed number that currently depends on `brake_diag.py`'s output, but the live
code would reproduce the same bug if it's run again.

**Second, independent gap: onset-timing frame order.** `brake_onset_ci.py` (L39),
`brake_timing.py` (L22), and `mechanism_all_configs.py` (L41) all build a run's reach-over-time
profile via `sorted(traj.items())`. `traj` comes straight from `json.load`, so its keys are
strings, and `sorted()` on `(str, value)` pairs sorts **lexicographically**, not numerically.
`closed_loop_harness/plan_flythrough.py` (L64) has to write `sorted(d.keys(), key=lambda
x:int(x))` to get chronological order — confirming `trajectories.json`'s keys are un-padded
numeric strings ("0", "1", ..., "23", ...), which a lexicographic sort scrambles for any rollout
with 10+ frames ("0","1","10","11",...,"19","2","20",...). Every onset-fraction number these three
scripts compute (`classify()`'s `i/len(profile)`, `brake_timing.py::onset()`, `load_run`'s onset
branch), and everything built on it — the paired onset-vs-FP16 deltas, the Spearman rho vs. NCAP
score — is therefore computed against a scrambled, non-chronological frame order for any run with
enough frames to hit two digits. The `max(profile)` frozen check and the frame-*count*-based
`early`/`MIN_FRAMES` classification are unaffected, since neither depends on order. This does not
currently corrupt a live claim: `FINDINGS.md`'s 2026-08-26 correction already voids "the
braking-onset mechanism" for the unrelated renderer-reuse reason, and no `papers/*.tex` file
mentions "onset" at all (grepped, zero matches) — so nothing in a paper rests on these numbers
today. But whenever the onset/mechanism analysis is redone under the restored harness, this
ordering bug needs a fix (sort by `int(key)`, not the raw string) first, or the redone numbers
will be wrong again for a second, unrelated reason.

**`RESULTS_clean_harness.md` vs. `compare_clean_vs_archived.py` — matches.** Traced `collect()`
(groups per-run scores by `(scene, category)` -> config) through the per-scenario table, the
per-config bootstrap over scenario-instance means, and the paired-difference bootstrap, against
the committed markdown: column headers/order, the aggregate CI values' methodology, and the
"ARCHIVED" column (a plain mean, no CI, and `--` for `w8` specifically because `benchmark12` — the
only archive `ARCH` points at — never had a w8 arm) all line up with what the script would
actually produce. No discrepancy found.

No `.tex` file touched (this task doesn't concern any `.tex` file). No `analysis/*.py` file
edited either: I did not have the underlying run data available in this sandbox to verify a fix
produces correct output, and per the mission ("if a result is surprising, stop and flag it... not
investigate further alone"), the two gaps above are flagged here for human review rather than
patched blind. `analysis/verify_before_commit.sh` passed (no `.tex` staged, so checks 1/3 skip; no
`RESULTS_clean_harness.md` change, so check 2 skips; no `.sh` changed). Only `AUTOLOOP.md` and this
`PROGRESS.md` entry are staged. Next unclaimed cloud-queue item: cross-check
`papers/four_bits_without_loss.tex`'s claims sentence-by-sentence against
`analysis/RESULTS_clean_harness.md`/`PROGRESS.md`/git log.

## 2026-08-31 — cross-check papers/four_bits_without_loss.tex claims against committed evidence — FLAGGED

Cloud routine cycle. Picked the next (and last remaining) unclaimed daily-queue item: sentence-
by-sentence cross-check of `papers/four_bits_without_loss.tex` against `analysis/RESULTS_clean_
harness.md`, `PROGRESS.md`, `FINDINGS.md`, `RESULTS.md`, and the git log. Read-only; no GPU/
renderer access used or needed.

**Traces cleanly (no issue):**
- Abstract + Table~\ref{tab:corrected}: FP16 4.81 [4.62,4.97], W8 4.81 [4.58,5.00], naive W4 4.71
  [4.38,5.00], group W4 g128 4.86 [4.67,4.98], and all three paired deltas vs FP16 — match
  `analysis/RESULTS_clean_harness.md` exactly (which itself was already verified against its
  generating script on 2026-08-30). Per-scene rows in the same table match
  `RESULTS_clean_harness.md`'s per-scenario table exactly, cell for cell.
- Table~\ref{tab:position} (scene 0099 stationary: FP16 5.00/5.00, naive W4 1.43/5.00, group W4
  0.00/5.00) matches commit `0791b63`'s message verbatim ("Scene 0099 stationary, 10 seeds --
  archived scores follow execution order"), and the "restored renderer" column matches
  `closed_loop_harness/run_sweep_clean.sh`'s own header comment ("Verified to reproduce the
  archive exactly (0099 stationary FP16 -> 5.0, 33 actors)").
- "168 backbone Linear layers (357.8M parameters)": matches `RESULTS.md:62-63`, `FINDINGS.md:37-
  38`, `analysis/W4_RECOVERY_PREP.md:10,50`, `strategy/claude.md:180`, and is computed (not just
  asserted) in `drivevla/inference_drivevla.py:154` and `orin/bench_planner_orin.py:144`
  (`n_params/1e6:.1f}M params ... in the Qwen backbone`).
- Table~\ref{tab:act} (activation quantization: FP16 0.595/42/0, W16A4 27.79(47x)/107/7, W16A4+DCT
  0.646/42/0, W4A4 9.34/127/3, W4A4+DCT 0.611/41/0) matches `icra27/e1/README.md` exactly, cell
  for cell, including the "47x" figure and the "42 (base rate)" framing.
- "16 scene/scenario instances x 10 seeds": `run_sweep_clean.sh`'s `CATS` dict sums to exactly 16
  instances across 12 scenes (0106/0108/0110/0278 each contribute 2 categories), `NRUNS` defaults
  to 10; matches `RESULTS_clean_harness.md`'s `n=16` and `run_sweep_clean.sh 10` invocation named
  in the paper's own provenance comment.
- "Within a single evaluation invocation multiple seeds are safe, since all seeds share the
  initialisation-time snapshot": correct per `run_sweep_clean.sh` L84-93 — actor restore happens
  once via `curl POST /update_actors` immediately before a single `main.py --runs "$NRUNS"`
  invocation, so all 10 seeds within that invocation do share one restored snapshot.
- The Section~5/Limitations claims about scope (LIBERO claim dropped, activation results are
  replay not closed-loop) match commit `b93e9d5` ("papers: scope to the driving VLA only, drop
  the manipulation/LIBERO claim") and `FINDINGS.md`'s "Unaffected" list.

**FLAGGED — real discrepancy, not just an untraceable claim.** Table~\ref{tab:corrected}'s "previously
reported" column gives W8 = **4.50**. But `analysis/RESULTS_clean_harness.md` — the exact file the
paper's own header comment cites as the source for "contaminated closed-loop" numbers
(`output/benchmark12`) — shows **`--`** for w8's ARCHIVED cell, not 4.50, and the 2026-08-30 cycle
already established why: benchmark12, the only archive `compare_clean_vs_archived.py` reads
ARCHIVED from, never had a w8 arm at all. I could not find 4.50 derived or asserted anywhere else
in committed history either (`RESULTS.md` has a single-seed W8 NCAP of 5.0 at scene 0103, not an
aggregate; no other file computes a w8 aggregate over the benchmark12 scenes). `FINDINGS.md`'s
correction-notice table (added in the same commit, `0791b63`) asserts the same 4.50, so this isn't
a copy error introduced later — it's been wrong (or at least unsourced) since the original
correction. This is exactly the kind of number this queue item exists to catch: a specific,
citable-looking figure with no traceable computation behind it, sitting right next to three other
numbers in the same table that do trace perfectly. Recommend a human check whether 4.50 comes from
some pre-benchmark12 run not committed to this repo, or whether it should be `--`/omitted like the
`RESULTS_clean_harness.md` cell it's supposed to mirror. Not fixed here per AUTOLOOP.md rule 1 (a
correction to what the table asserts is a claim change, not a mechanical LaTeX fix) — flagged only.

**FLAGGED — untraceable narrative claims (lower severity, no contradiction found, just no
source).** Two specific-sounding numeric claims in Section~\ref{sec:mech}/\ref{sec:corrected} have
no committed backing anywhere I could find (`output/` is gitignored, so the raw logs behind them,
if they exist, were never committed):
1. "Measured on scene 0099: a freshly loaded renderer reports 28 actors; after one evaluation it
   reports 1; a subsequent evaluation initialised from that mutated state logged 60 actors,
   against 33 in the reference run." Only "33" (the pristine/restored count) is corroborated,
   by `run_sweep_clean.sh`'s header comment; 28, 1, and 60 appear nowhere else in git history,
   including the original correction commit `0791b63`'s message, which gives the NCAP-score
   version of this same scene/mechanism but not actor counts.
2. "Cost is negligible---one HTTP request---against the 70--100\,s of reloading the renderer."
   `run_sweep_clean.sh`'s `wait_up` polls up to 90x5s=450s worst case and stagger is 100s per
   lane, but I found no committed measurement giving "70-100s" as the actual observed renderer
   load time specifically.
These are narrative texture, not headline results, and neither contradicts anything else in the
repo — flagging per the letter of the queue item ("flag any claim that isn't traceable to
committed evidence"), not because I believe them wrong.

No `papers/*.tex` file was edited (per rule 1 — these are findings, not mechanical LaTeX fixes,
even the W8 discrepancy). `analysis/verify_before_commit.sh` passed (no `.tex` staged, so checks
1/3 skip; no `RESULTS_clean_harness.md` change, so check 2 skips; no `.sh` changed). Only
`AUTOLOOP.md` and this `PROGRESS.md` entry are staged. This was the last unclaimed item in the
daily cloud routine queue — the queue is now `[x]` in full except items requiring GPU access
(interactive/local-only queue, out of scope for this loop).

## 2026-09-21 13:21 — daily cloud routine — NO-OP (empty queue)

Cloud routine cycle. Checked out `auto/opendrivevla-verify` (already existed on origin, so no new
branch created), fetched clean, read `AUTOLOOP.md` in full and this file's tail. Every item in
the "Daily cloud routine queue" section of `AUTOLOOP.md` is already `[x]` (last one closed
2026-08-31, confirmed by the entry immediately above this one). The only remaining unclaimed
items are in the "Interactive/local-only queue" (the `run_freedrive_clean.sh` re-run and the
closed-loop CI extension), both of which explicitly require GPU access this cloud sandbox does
not have (rule 5) — no matching queue item for this execution mode.

Per AUTOLOOP.md's "Every cycle ends with" section ("If nothing was accomplished ... no matching
queue item for this execution mode ... log that too"), this is a logged no-op rather than an
invented task. No files other than this `PROGRESS.md` entry were touched — `AUTOLOOP.md` needs no
edit since its queue state already correctly reflects `[x]` on every cloud-eligible item.
`analysis/verify_before_commit.sh` gate run before commit (see below): no `.tex` staged, so checks
1/3 skip; no `RESULTS_clean_harness.md` change, so check 2 skips; no `.sh` changed, so checks 4/5
skip; gate passed trivially. Nothing for a human to act on here — this entry exists only so a
silent cycle doesn't look like the loop died, per AUTOLOOP.md's own reasoning. The cloud queue
stays empty until a human adds a new cloud-eligible item or unblocks one of the two GPU-dependent
interactive items for local execution.
