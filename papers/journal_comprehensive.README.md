# Journal — Comprehensive Study

**File:** `journal_comprehensive.tex` · **Venue:** journal · **Status:** draft (unpublished)
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## Thesis
The full treatment: (1) open-loop L2 is blind (19× ego-ablation); (2)+(3) closed-loop reveals a
per-channel INT4 generation collapse that group-wise INT4 recovers (granularity, not bit-width);
(4) component memory benchmarks + edge analysis (planner free, perception is the wall). Explicit
threats-to-validity throughout.

## Key results (real-camera NeuroNCAP, 14 scenes / 16 scenario instances, 10 seeds)
Mean generation coherence (well-formed predicted-frame rate):

| Config | mean coherence | per-scenario behaviour |
|--------|----------------|------------------------|
| FP16 | 76.0% | — |
| INT4 per-channel | **3.7%** | collapses (≤15%) on 15/16 |
| INT4 group-128 | **73.5%** | recovers (≥85% of FP16) on 13/16 |

Full 16-row per-scene table (coherence) is in the `.tex`. Group-wise ties/beats FP16 on 13/16;
partial on 3 (0099, 0101, 0796). Collision/NCAP confounded: several scenes fail 10/10 for every config
including FP16 (e.g. 0278_stationary, 0796_stationary). Open-loop: INT4 0.33 m; ego-ablation 6.38 m
(19×). Backbone footprint: ~0.26 GB at INT4 g128.

## Scope / caveats (Threats to Validity section)
14 scenes / 16 scenario instances; OOD renderer (several scenes failed even by FP16); partial
group-wise recovery on 0099/0101/0796; collision/NCAP
confounded (lead with coherence); INT3/INT2/AWQ only on the earlier blank-camera renderer; fake-quant;
edge throughput rows are estimates.

## Path to submission
Most complete of the five but needs the same conference-gating items (more scenes — 12 staged —,
perception-validity fix, standard NCAP reporting, mechanism ablation) plus real bibliography.

## Build
`latexmk -pdf journal_comprehensive.tex`. Fill bib placeholders before submission.
