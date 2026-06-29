# Papers

Five manuscripts from the OpenDriveVLA quantization study. All share four authors — Justin Williams,
Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University) — and **unpublished draft**
status. Each `.tex` has a matching `*.README.md` with its thesis, target venue, key numbers, and
build notes.

| File | Venue | One-line thesis |
|------|-------|-----------------|
| `workshop1_openloop_blind.tex` | Workshop | Open-loop nuScenes L2 is ~ego-extrapolation (19× ablation), so "INT4 is lossless" is a metric artifact. |
| `workshop2_granularity.tex` | Workshop | Scale **granularity**, not bit-width, decides whether quantized generation survives domain shift. |
| `workshop3_edge_memory.tex` | Workshop | The planner is free for the edge; **perception** is the memory/latency wall. |
| `conference_closedloop_reveals.tex` | Main conference | Open-loop hides quantization damage; closed-loop reveals a granularity collapse with a near-free fix. |
| `journal_comprehensive.tex` | Journal | The full study: blindness + collapse + granularity fix + edge memory, with threats-to-validity. |

## Headline numbers (real-camera NeuroNCAP, 2 scenes, 10 seeds each)

Generation coherence (well-formed predicted-frame rate):

| Config | 0103 frontal | 0796 stationary |
|--------|--------------|-----------------|
| FP16 | 74.3% | 72.4% |
| INT4 per-channel | 0.0% | 8.6% |
| INT4 group-128 | 72.9% | 57.9% |

Open-loop: INT4 L2 = 0.33 m (lossless); ego-state ablation → 6.38 m (**19×**).

## Building

No system TeX on the research box. Compile with `latexmk -pdf <file>.tex` (or Overleaf). Each paper is
self-contained `article` class (booktabs, authblk, hyperref). Bibliographies are inline
`thebibliography`; some entries still carry `(Replace with verified citation.)` placeholders to fill
before submission.

## Status / caveats
Draft. Results are two-scene; group-wise recovery is partial on the harder scene; one scene (0796) is
failed by every config including FP16 (collision metric is confounded — coherence is the lead metric).
See `../RESULTS.md` and `../FINDINGS.md`.
