# Journal — Comprehensive Study

**File:** `journal_comprehensive.tex` · **Venue:** journal · **Status:** draft (unpublished)
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## Thesis
The full treatment: (1) open-loop L2 is blind (19× ego-ablation); (2)+(3) closed-loop reveals a
per-channel INT4 generation collapse that group-wise INT4 recovers (granularity, not bit-width);
(4) component memory benchmarks + edge analysis (planner free, perception is the wall). Explicit
threats-to-validity throughout.

## Key results (real-camera NeuroNCAP, 2 scenes, 10 seeds)
| Config | 0103 coher. | 0796 coher. | 0103 NCAP/coll | 0796 NCAP/coll |
|--------|-------------|-------------|----------------|----------------|
| FP16 | 74.3% | 72.4% | 5.0, 0/10 | 0.88, 10/10 |
| INT4 per-channel | 0.0% | 8.6% | 4.08, 4/10 | 0.00, 10/10 |
| INT4 group-128 | 72.9% | 57.9% | 5.0, 0/10 | 1.58, 10/10 |

Open-loop: INT4 0.33 m; ego-ablation 6.38 m (19×). Backbone footprint: ~0.26 GB at INT4 g128.

## Scope / caveats (Threats to Validity section)
Two scenes; OOD renderer (FP16 fails 0796); partial group-wise recovery on 0796; collision/NCAP
confounded (lead with coherence); INT3/INT2/AWQ only on the earlier blank-camera renderer; fake-quant;
edge throughput rows are estimates.

## Path to submission
Most complete of the five but needs the same conference-gating items (more scenes — 12 staged —,
perception-validity fix, standard NCAP reporting, mechanism ablation) plus real bibliography.

## Build
`latexmk -pdf journal_comprehensive.tex`. Fill bib placeholders before submission.
