#!/usr/bin/env python3
"""make_config.py OUT.json [--prefix-caching 0|1] [--notes-file F] — config.json for the vLLM run dir (keys per T8 section_vllm.py)."""
import json, os, subprocess, sys, datetime, glob
ROOT="/home/luk/dev/airun/vllm-pp6"; T7=os.environ.get("T7DIR", f"{ROOT}/work/T7")
out=sys.argv[1]; pc=("--prefix-caching" in sys.argv and sys.argv[sys.argv.index("--prefix-caching")+1]=="1")
notes=[]
if "--notes-file" in sys.argv: notes=[l.strip() for l in open(sys.argv[sys.argv.index("--notes-file")+1]) if l.strip()]
docker_run=""
for l in open(f"{T7}/vllm-server.log", errors="replace"):
    if l.startswith("# docker"): docker_run=l[2:].strip(); break
img="vllm-pp6:7d0b4e57a"
digest=subprocess.run(["docker","image","inspect",img,"--format","{{.Id}}"],capture_output=True,text=True).stdout.strip()
ov=f"{ROOT}/work/T1-patches/overlay"
overlay=sorted(os.path.relpath(p,f"{ROOT}/work/T1-patches") for p in glob.glob(f"{ov}/**/*.py",recursive=True))
ck="/home/luk/dev/ai/cache/qwen38-flash-next-gptq4"
files=sorted([n,os.path.getsize(f"{ck}/{n}")] for n in os.listdir(ck) if os.path.isfile(f"{ck}/{n}"))
env={}
for tok in docker_run.split():
    pass
import re
for m in re.finditer(r"-e ([A-Z_]+=[^ ]+)", docker_run.replace("\\=","=")):
    k,v=m.group(1).split("=",1); env[k]=v.strip("'\"").replace("\\,",",")
cfg={
 "docker_run": docker_run or "TBD (work/T7/vllm-server.log header)",
 "image": img, "image_digest": digest, "image_base": "vllm/vllm-openai:cu129-nightly @ 7d0b4e57a (torchcodec removed)",
 "vllm_version": "0.31.1rc1.dev253+g7d0b4e57a", "vllm_commit": "7d0b4e57a", "torch": "2.13.0+cu129",
 "patches": ["overlay.diff in this directory (unified diff of the 7 overlaid files against the image originals)"],
 "overlay_files": overlay, "env": env,
 "part": "VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8 (pipeline_parallel_size 6, tensor_parallel_size 1, mp executor)",
 "max_model_len": 262144, "max_num_seqs": 5, "kv_cache_dtype": "auto (bf16)", "prefix_caching": pc,
 "engram_config": {"cpu_offload": True},
 "checkpoint": {"repo": "btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit", "local": ck, "files": files},
 "measured": f"{datetime.date.today().isoformat()} (grid-vllm.json)",
 "gpus": "6 x RTX 3090 (GPUs 0-5, one pipeline stage each; GPU 6 RTX 5060 Ti unused), AMD 7950X, 187 GB RAM",
 "notes": notes,
}
json.dump(cfg,open(out,"w"),indent=2); print("wrote",out,"overlay files:",len(overlay),"ckpt files:",len(files))
