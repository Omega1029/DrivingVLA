# Papers

Manuscripts from the OpenDriveVLA quantization study. All share four authors — Justin Williams,
Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University) — and **unpublished draft**
status. Each `.tex` has a matching `*.README.md` with its thesis, target venue, key numbers, and
build notes.

**`opendrivevla_full_paper.tex` is the combined, comprehensive paper** — all evaluations
(open-loop, closed-loop 14-scene, on-device Orin systems + accuracy + per-scene success), the
quantization methodology (RTN / per-channel vs group-wise / the outlier–scale mechanism), and
comparison tables to prior planners (UniAD/VAD/OpenDriveVLA) and quantizers (RTN/GPTQ/AWQ). The five
drafts below remain for venue-specific submission.

| File | Venue | One-line thesis |
|------|-------|-----------------|
| `opendrivevla_full_paper.tex` | **Combined** | **Everything: blindness + granularity fix + edge + on-Orin systems/accuracy/per-scene, with methodology and prior-method comparisons.** |
| `workshop1_openloop_blind.tex` | Workshop | Open-loop nuScenes L2 is ~ego-extrapolation (19× ablation), so "INT4 is lossless" is a metric artifact. |
| `workshop2_granularity.tex` | Workshop | Scale **granularity**, not bit-width, decides whether quantized generation survives domain shift. |
| `workshop3_edge_memory.tex` | Workshop | The planner is free for the edge; **perception** is the memory/latency wall. |
| `conference_closedloop_reveals.tex` | Main conference | Open-loop hides quantization damage; closed-loop reveals a granularity collapse with a near-free fix. |
| `journal_comprehensive.tex` | Journal | The full study: blindness + collapse + granularity fix + edge memory, with threats-to-validity. |

## Headline numbers (real-camera NeuroNCAP, 14 scenes / 16 scenario instances, 10 seeds each)

Mean generation coherence (well-formed predicted-frame rate):

| Config | mean coherence | per-scenario behaviour |
|--------|----------------|------------------------|
| FP16 | 76.0% | — |
| INT4 per-channel (naive) | **3.7%** | collapses (≤15%) on 15/16 |
| INT4 group-128 | **73.5%** | recovers (≥85% of FP16) on 13/16 |

Group-wise ties/beats FP16 on 13/16 scenarios; partial recovery on 3 (0099, 0101, 0796).
Open-loop: INT4 L2 = 0.33 m (lossless); ego-state ablation → 6.38 m (**19×**).

## Building

No system TeX on the research box. Compile with `latexmk -pdf <file>.tex` (or Overleaf). Each paper is
self-contained `article` class (booktabs, authblk, hyperref). Bibliographies are inline
`thebibliography`; some entries still carry `(Replace with verified citation.)` placeholders to fill
before submission.

## Status / caveats
Draft. Results span 14 scenes (16 scenario instances) — the full released closed-loop benchmark;
group-wise recovery is partial on 3 (0099, 0101, 0796); several scenes are failed by every config
including FP16 (collision metric is confounded — coherence is the lead metric).
See `../RESULTS.md` and `../FINDINGS.md`.
