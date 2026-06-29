#!/usr/bin/env python
"""On-device (Jetson Orin) benchmark of the OpenDriveVLA-0.5B PLANNER head.

Measures the quantized component (the Qwen2.5-0.5B language/action backbone) only:
this needs just torch + transformers, NOT the mmcv/mmdet3d perception ops, so it runs
on a freshly flashed Orin without building UniAD. Reports FP16 vs W4(group-128) and
W8 on the SAME device, plus the *packed* INT4 footprint (the real memory win).

Run on the Orin:
    python orin/bench_planner_orin.py --ckpt checkpoints/OpenDriveVLA-0.5B \
        --seqlen 1024 --gen 60 --iters 30 --out orin_planner_bench.json

Honesty notes baked into the output:
  * runtime_quant = "fake" : weights are quantized then dequantized to the compute dtype
    (this repo has no packed-INT4 GEMM), so LATENCY reflects FP16 math. We therefore report
    latency as a *parity* check and report MEMORY two ways: (a) runtime (fake-quant, ~FP16)
    and (b) packed_bytes = the true on-disk/in-RAM size if stored as packed INTb + FP16 scales.
"""
import argparse, json, time, os, sys, gc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.quantizers import _quantize_weight, _iter_backbone_linears


def device_info():
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    # Jetson model
    try:
        with open("/proc/device-tree/model") as f:
            info["board"] = f.read().strip("\x00").strip()
    except Exception:
        info["board"] = "unknown (not a tegra?)"
    try:
        with open("/etc/nv_tegra_release") as f:
            info["jetpack"] = f.readline().strip()
    except Exception:
        pass
    return info


def backbone_linear_stats(model):
    n_params = 0
    for m in _iter_backbone_linears(model):
        n_params += m.weight.numel()
    return n_params


def packed_bytes(n_params, bits, group):
    """True footprint if weights were stored packed: n_params*bits/8 + FP16 scales."""
    w = n_params * bits / 8.0
    if group and group > 0:
        n_scales = n_params / group
    else:
        n_scales = n_params / 1.0  # per-channel ~ one scale per output row; approx with /in but
        # report conservative per-group=channel; dominated by weight bytes anyway
        n_scales = 0  # per-channel scales negligible vs weights
    scales = n_scales * 2  # FP16 scale per group
    return w + scales


@torch.no_grad()
def apply_fakequant(model, bits, group):
    for m in _iter_backbone_linears(model):
        wf = m.weight.data.float()
        wq = _quantize_weight(wf, bits, group, asym=False)
        m.weight.data.copy_(wq.to(m.weight.dtype))


@torch.no_grad()
def bench_forward(model, seqlen, gen, iters, dev, vocab):
    """Time a representative planner generate: prefill seqlen, decode `gen` tokens."""
    torch.cuda.reset_peak_memory_stats() if dev.startswith("cuda") else None
    ids = torch.randint(0, vocab, (1, seqlen), device=dev)
    # Force EXACTLY `gen` new tokens for every config (min==max, ignore EOS) so latency is a
    # fair fixed-work comparison — otherwise an early EOS on random input fakes a speedup.
    gkw = dict(min_new_tokens=gen, max_new_tokens=gen, do_sample=False,
               use_cache=True, pad_token_id=0, eos_token_id=None)
    for _ in range(3):  # warmup
        out = model.generate(ids, **gkw)
    n_new = out.shape[1] - seqlen
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.generate(ids, **gkw)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    t.sort()
    res = {"median_s": t[len(t)//2], "min_s": t[0], "max_s": t[-1],
           "n_new_tokens": int(n_new), "tok_per_s": n_new / t[len(t)//2]}
    if dev.startswith("cuda"):
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--seqlen", type=int, default=1024)   # ~ perception tokens + prompt
    ap.add_argument("--gen", type=int, default=60)        # trajectory text tokens
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="orin_planner_bench.json")
    args = ap.parse_args()

    import json as _json
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from safetensors.torch import load_file
    dev = args.device
    dtype = getattr(torch, args.dtype)

    report = {"device": device_info(), "args": vars(args), "runs": {}}
    print("device:", json.dumps(report["device"], indent=2))

    # Build a pure Qwen2.5-0.5B decoder from the LlavaQwen config (drop mm_*/vision fields),
    # then load ONLY the decoder weights (model.layers/embed/norm + lm_head). This is the exact
    # quantized planner component; vision tower & projectors are not on the planner's hot path.
    t0 = time.perf_counter()
    raw = _json.load(open(os.path.join(args.ckpt, "config.json")))
    keep = {k: raw[k] for k in raw if not (k.startswith("mm_") or "vision" in k or k == "architectures")}
    qcfg = Qwen2Config(**{k: v for k, v in keep.items() if k in Qwen2Config().to_dict()})
    qcfg.torch_dtype = args.dtype
    model = Qwen2ForCausalLM(qcfg).to(dtype)
    sd = load_file(os.path.join(args.ckpt, "model.safetensors"))
    want = set(model.state_dict().keys())
    filt = {k: v for k, v in sd.items() if k in want}
    miss = model.load_state_dict(filt, strict=False)
    model.to(dev).eval()
    load_s = time.perf_counter() - t0
    vocab = qcfg.vocab_size
    report["weights_loaded"] = len(filt)
    report["weights_missing"] = len(miss.missing_keys)
    print(f"loaded {len(filt)} decoder tensors ({len(miss.missing_keys)} missing) from real checkpoint")
    n_params = backbone_linear_stats(model)
    report["load_s"] = load_s
    report["backbone_linear_params"] = n_params
    print(f"loaded in {load_s:.1f}s | backbone Linear params = {n_params/1e6:.1f}M | vocab={vocab}")

    fp_weights = [m.weight.data.clone() for m in _iter_backbone_linears(model)]

    def restore():
        for m, w in zip(_iter_backbone_linears(model), fp_weights):
            m.weight.data.copy_(w)

    configs = [("fp16", 0, 0), ("int8_g128", 8, 128), ("w4_g128", 4, 128), ("w4_perchannel", 4, 0)]
    for name, bits, group in configs:
        restore()
        if bits > 0:
            apply_fakequant(model, bits, group)
        gc.collect(); torch.cuda.empty_cache() if dev.startswith("cuda") else None
        r = bench_forward(model, args.seqlen, args.gen, args.iters, dev, vocab)
        if bits == 0:
            r["runtime_quant"] = "none"
            r["weight_bytes_fp16"] = n_params * 2
            r["weight_gb"] = n_params * 2 / 1e9
        else:
            r["runtime_quant"] = "fake (dequant to %s)" % args.dtype
            r["packed_weight_bytes"] = packed_bytes(n_params, bits, group)
            r["packed_weight_gb"] = packed_bytes(n_params, bits, group) / 1e9
        report["runs"][name] = r
        print(f"  {name:14s} median {r['median_s']*1000:7.1f} ms  "
              f"{r['tok_per_s']:6.1f} tok/s  peak {r.get('peak_mem_gb', float('nan')):.2f} GB")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)

    # paper-ready table
    print("\n=== PAPER TABLE (planner only, on", report["device"].get("board", "?"), ") ===")
    print("config         | latency (ms) | tok/s | runtime peak (GB) | packed weights (GB)")
    for name, r in report["runs"].items():
        pk = r.get("packed_weight_gb", r.get("weight_gb", float("nan")))
        print(f"{name:14s} | {r['median_s']*1000:11.1f} | {r['tok_per_s']:5.1f} | "
              f"{r.get('peak_mem_gb', float('nan')):16.2f} | {pk:.3f}")


if __name__ == "__main__":
    main()
