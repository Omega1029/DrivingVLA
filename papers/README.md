# Papers

Three manuscripts from the OpenDriveVLA quantization study, each aimed at a target venue. All share
four authors — Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta
University) — and **unpublished draft** status.

| File | Target venue | One-line thesis |
|------|-------|-----------------|
| `workshop1_openloop_blind.tex` | **CoRL 2026** | Open-loop nuScenes L2 is ~ego-extrapolation (19× ablation), so "INT4 is lossless" is a metric artifact; certify closed-loop. |
| `workshop2_granularity.tex` | **IROS 2026** | Scale **granularity**, not bit-width, decides whether quantized generation survives domain shift. |
| `journal_comprehensive.tex` | **NeurIPS 2026** | The full study: blindness + collapse + granularity fix + edge memory (incl. measured Jetson Orin), with threats-to-validity. |

> **Consolidation note:** `conference_closedloop_reveals.tex` (closed-loop reveal) and
> `workshop3_edge_memory.tex` (edge memory) were merged into `journal_comprehensive.tex`, which is a
> superset — the closed-loop collapse/recovery results and the measured on-device Jetson AGX Orin
> table now live there.

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
