Short answer: **yes — it's in scope, and the timing lines up well.** ICRA 2027 takes place in Seoul, May 24–28, 2027, with the first submission deadline on September 15, 2026 (11:59 PST) via PaperPlaza — about two months from now. Here's the full picture.

## Logistics you need to know (from the official CFP)

- **Page limit:** 8 pages total, including all material and references — supplementary material must fit within the 8 pages, though a video attachment is allowed. Your draft looks like it'll land around 7–8 pages compiled, so you're probably fine but should check.
- **Double-anonymous review:** manuscripts must exclude authors and affiliations. Your current LaTeX has the full author block, and your Reproducibility section references "the public release" — you'll need an anonymized repo link (e.g., anonymous.4open.science).
- You must select at least three ICRA keywords at submission — good fits would be Autonomous Vehicle Navigation, Deep Learning Methods, and Embedded Systems for Robotics/Automation.
- ICRA 2027 is on-site only; no-show papers are skipped, and any AI-generated content must be disclosed in the acknowledgments.
- There's an "IROS-ICRA transfer" category for papers previously rejected from IROS, requiring a response file addressing prior reviews — relevant only if a version of this went to IROS 2026.
- **Competition context:** ICRA 2026 received a record 5,088 submissions, and ICRA 2025 accepted 1,606 of 4,153 papers (~38.7%). Expect 2027 to be at least as competitive.

## Fit, read through the Reuss blog post

The blog is actually quite favorable to your positioning:

**You clear his VLA definition.** His criterion is a pretrained backbone trained on large-scale vision-language data, subsequently trained to generate control commands — and he explicitly lists "steering angle for a car" as a qualifying output. OpenDriveVLA (Qwen2.5 backbone → trajectory text) qualifies cleanly, so nobody can dismiss this as "just a multimodal policy" paper.

**You sit at the intersection of two of his nine trends.** "Efficient VLAs" (quantization, distillation for inference) and "Evaluation and Benchmarking of VLAs" are both identified as active research categories. More importantly, your central thesis is the driving-domain version of his core critique: he argues that saturated benchmarks where scores cluster near the ceiling mask real progress, that sim-only results are hard to trust, and that current benchmarks are a poor proxy for robust behavior in messy environments. Your "open-loop L2 is blind, closed-loop reveals the damage" finding is exactly this argument, made rigorous with the 19× ego-ablation. That framing is fashionable right now — lean into it in the intro.

**ICRA rewards your hardware component more than ICLR would.** The blog notes real-world results are what make VLA claims trustworthy, and ICRA reviewers specifically value deployment on actual robot-grade hardware. The Orin section is a differentiator at this venue in a way it wouldn't be at a pure ML conference.

## What the blog reveals you must fix before submitting

This is the biggest issue: **your related work is missing the VLA quantization literature, and reviewers from this community will know it.** The blog highlights AutoQVLA ("Not All Channels Are Equal in Vision-Language-Action Model's Quantization"), an ICLR 2026 submission that analyzes quantization of OpenVLA and proposes an improved method maintaining performance at 30% of the original VRAM. Your intro says VLA quantization "has been studied" on robot arms, but your bibliography cites none of it. You need to cite AutoQVLA and do a sweep for other manipulation-VLA quantization work (there's been a cluster of it — low-bit and saliency-aware quantization for VLAs), then scope your "first" claim precisely: first *closed-loop* and first *driving-domain* characterization. That claim survives the comparison — AutoQVLA is manipulation, open-loop-style benchmarks — but only if you make the distinction explicitly rather than leaving reviewers to catch the omission.

Other likely reviewer pressure points, in rough order of risk:

1. **RTN-only closed-loop.** Your isolation argument (vary only granularity) is sound, but AWQ/GPTQ are the deployed defaults, and you admit AWQ/INT3/INT2 were only characterized in the discarded pre-renderer-fix setup. Budget compute for at least AWQ group-128 through the fixed closed-loop harness — it's the single most predictable reviewer request.
2. **The coherence metric.** A skeptic will argue malformed-text collapse could be patched with constrained decoding or a retry fallback, making it an engineering artifact rather than a safety finding. Preempt this with a short experiment or argument showing the representational damage persists under constrained decoding.
3. **Single model, single scale.** The blog community is sensitized to overfitting claims to one setup. Even a partial 3B replication of the granularity effect (or an honest limitation paragraph explaining why 0.5B is the edge-relevant scale) would help.
4. **Prior-version overlap.** Your LaTeX comments say this merges three workshop papers, a conference paper, and a journal draft. PaperPlaza runs similarity checks; if any of those are on Xplore or arXiv, rewrite overlapping text and cite them. And if the journal draft is simultaneously under review anywhere with the same material, that's a dual-submission violation — resolve it before September 15.
5. **Flat latency.** You're honest that fake-quant yields no speedup, but make sure no table caption or abstract phrasing implies a measured latency win; ICRA systems reviewers read those tables closely.

One tactical suggestion: submit a **supplementary video**. You already have closed-loop video renderers — a side-by-side of FP16 vs. per-channel INT4 (collapsing) vs. group-128 in the same NeuroNCAP scenario would be the most persuasive 60 seconds in your submission, and video is a norm at ICRA in a way it isn't at ML venues.

So: scope fit is strong, the blog's framing actively supports your thesis, and two months is enough to fix anonymization, the related-work gap, and ideally one AWQ closed-loop run. Want me to draft the anonymized version, write the related-work paragraph situating you against AutoQVLA and the manipulation-quantization line, or restructure the intro around the benchmark-blindness framing?
