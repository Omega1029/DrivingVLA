#!/usr/bin/env python
"""Replay captured planner inputs (fused inputs_embeds) through the OpenDriveVLA-0.5B planner at a
chosen quantization, emitting a plan_conv json that drivevla/eval_drivevla.py scores for open-loop L2.

Runs identically on A100 (reference) and Jetson Orin (deployment) -> isolates device + quantization.
Needs only torch + transformers (decoder only; no perception / mmcv).

    python orin/replay_accuracy.py --ckpt checkpoints/OpenDriveVLA-0.5B \
        --embeds-dir embeds --bits 4 --group 128 --mode rtn_sym --out plan_conv_w4g128.json
"""
import argparse, os, sys, glob, json, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.quantizers import _quantize_weight, _iter_backbone_linears


def load_planner(ckpt, dtype, dev):
    import json as _json
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from safetensors.torch import load_file
    raw = _json.load(open(os.path.join(ckpt, "config.json")))
    keep = {k: raw[k] for k in raw if not (k.startswith("mm_") or "vision" in k or k == "architectures")}
    qcfg = Qwen2Config(**{k: v for k, v in keep.items() if k in Qwen2Config().to_dict()})
    model = Qwen2ForCausalLM(qcfg).to(dtype)
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    want = set(model.state_dict().keys())
    miss = model.load_state_dict({k: v for k, v in sd.items() if k in want}, strict=False)
    model.to(dev).eval()
    return model, len(want) - len(miss.missing_keys)


@torch.no_grad()
def apply_fakequant(model, bits, group):
    if bits <= 0:
        return
    for m in _iter_backbone_linears(model):
        wf = m.weight.data.float()
        m.weight.data.copy_(_quantize_weight(wf, bits, group, asym=(False)).to(m.weight.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--embeds-dir", required=True)
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--mode", default="rtn_sym")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    dtype = getattr(torch, args.dtype)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)

    model, nload = load_planner(args.ckpt, dtype, dev)
    apply_fakequant(model, args.bits, args.group)
    print(f"[replay] {args.bits}b g{args.group} | loaded {nload} decoder tensors | device {dev}")

    files = sorted(glob.glob(os.path.join(args.embeds_dir, "*.pt")))
    t0 = time.time()
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for fp in files:
            d = torch.load(fp, map_location=dev)
            emb = d["inputs_embeds"].to(dev, dtype)
            am = d["attention_mask"]
            am = am.to(dev) if am is not None else None
            pid = d.get("position_ids")
            pid = pid.to(dev) if pid is not None else None
            gkw = dict(inputs_embeds=emb, attention_mask=am, do_sample=False, num_beams=1,
                       max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            if pid is not None:
                gkw["position_ids"] = pid
            out = model.generate(**gkw)
            text = tok.batch_decode(out, skip_special_tokens=True)[0]
            f.write(json.dumps({"id": d["id"], "question": "", "answer": [text]},
                               ensure_ascii=False) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"  {n}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    print(f"[replay] wrote {n} -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
