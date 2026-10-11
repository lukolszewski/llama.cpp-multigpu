# vllm-pp6: upstream nightly pinned by digest, minus torchcodec.
# The cu129 nightly ships a torchcodec wheel linked against libnvrtc.so.13, which
# the image does not have; vllm.multimodal.media.audio only guards ImportError /
# RuntimeError, so `vllm serve` dies with OSError at import. Removing the package
# turns that into the guarded ImportError (audio decoding is not used here).
FROM vllm/vllm-openai:cu129-nightly@sha256:d9af02a8ff61eb28fcadff123af9351cdfc3948f451b5e9dc4eb772aa80c9ee6
# upstream build: VLLM_BUILD_COMMIT=7d0b4e57aac4c4323225b0eee9ae960ce75ffffe, vllm 0.31.1rc1.dev253+g7d0b4e57a
RUN pip uninstall -y torchcodec
