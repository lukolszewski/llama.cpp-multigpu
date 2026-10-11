03:05:09 steps.sh START (T7 grid) PREFIX_CACHE=0 NSEQ=5 SIZES=5000,50000,150000,200000,250000
               total        used        free      shared  buff/cache   available
Mem:             187          28          68           1          92         159
>> log: /home/luk/dev/airun/vllm-pp6/work/T7/vllm-server.log
docker run -d --name vllm-pp6 --gpus \"device=0\,1\,2\,3\,4\,5\" --ipc=host --shm-size 16g -p 8010:8010 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e VLLM_PP_LAYER_PARTITION=8\,8\,8\,8\,8\,8 -v /home/luk/dev/ai/cache/qwen38-flash-next-gptq4:/model:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/model_executor/models/config.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/config.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/amd/model_state.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/model_state.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/common/ngram_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/ngram_embedding.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/nvidia/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/nvidia/model_state.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model_state.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/nvidia/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/v1/core/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro vllm-pp6:7d0b4e57a /model --served-model-name qwen3.8-flash-next --dtype bfloat16 --pipeline-parallel-size 6 --tensor-parallel-size 1 --distributed-executor-backend mp --max-model-len 262144 --max-num-seqs 5 --gpu-memory-utilization 0.92 --engram-config \{\"cpu_offload\":\ true\} --port 8010 --host 0.0.0.0 --trust-remote-code --no-enable-prefix-caching 
f991a77618d371fd2138f94839f3b64adc08581d6fccba90345e47629fa83df5
>> started vllm-pp6 on port 8010 (logs → /home/luk/dev/airun/vllm-pp6/work/T7/vllm-server.log)
03:11:26 healthy after 376s
03:11:28 grid phase 1: 1 session, sizes 5000,50000,150000,200000,250000
03:12:53 phase 1 exit=0
03:12:53 1-session 250k pp_agg=9046.1 (abort threshold 262.7/3)
03:12:53 grid phase 2: 5 sessions, sizes 5000,50000,150000,200000,250000
03:19:43 phase 2 exit=0
03:19:45 error-line count vllm-server.log: 0
03:19:45 stopping vLLM
Error response from daemon: Could not kill running container f991a77618d371fd2138f94839f3b64adc08581d6fccba90345e47629fa83df5, cannot remove - container f991a77618d3 PID 3537628 is zombie and can not be killed. Use the --init option when creating containers to run an init inside the container that forwards signals and reaps processes
>> stopped vllm-pp6
03:19:55 steps.sh END
