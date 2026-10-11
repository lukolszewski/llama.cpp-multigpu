# Tooling used for this run (machine-01 specific, kept for reproduction and amendments)

- `image/Dockerfile` — the `vllm-pp6:7d0b4e57a` image: upstream nightly pinned by digest, torchcodec removed.
- `run-vllm-pp6.sh [stop]` — starts/stops the vLLM container with the 7 overlay files bind-mounted
  (`OVERLAY=` dir), PP=6 partition validation, all knobs as env vars. The overlay files are the
  `pp-nvidia-qwen` branch of https://github.com/lukolszewski/vllm (`overlay.diff` here is the same change).
- `window.sh LABEL CMD...` — GPU window wrapper: stops the production llama.cpp server through its
  launcher, runs CMD under a timeout, always stops vLLM and restores production (verified on /health).
- `steps.sh` — the T7 grid procedure run inside the window: start vLLM, wait for /health (20 min abort),
  snapshot nvidia-smi/docker stats/free, run `readme_grid_openai.py` for 1 and 5 sessions, merge JSON.
- `make_config.py` / `section_vllm.py` — build `config.json` from the server log header and render the
  benchmarks.md section + 7-column table from the three grid JSON files.

`../OVERLAY-NOTES.md` explains every overlay change (root cause, where it was found) and which patches of
the earlier alanoo81/flashnext-v620-pp3 fork are obsolete on this vLLM commit.
