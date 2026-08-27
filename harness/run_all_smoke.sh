#!/usr/bin/env bash
# Step 1L across the remaining model list, one at a time.
# Weights are freed after each model so a 150GB container disk suffices.
export PATH=/workspace/venv/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -f /workspace/.hf_token ] && export HF_TOKEN=$(cat /workspace/.hf_token)

cd /workspace/harness
PY=/workspace/venv/bin/python

run() {                       # run <repo> [extra args...]
  local repo="$1"; shift
  local slug=${repo//\//_}
  echo "=================================================================="
  echo ">>> $repo   $(date -u +%H:%M:%S)"
  df -h / | tail -1
  $PY gpu_smoke_test.py --model "$repo" "$@" > "/workspace/smoke_${slug}.log" 2>&1
  local rc=$?
  echo "exit=$rc"
  sed -n '/^{/,/^}/p' "/workspace/smoke_${slug}.log" | head -40
  [ $rc -ne 0 ] && grep -E "Error|error:|ValueError|RuntimeError|Exception" \
      "/workspace/smoke_${slug}.log" | grep -v Traceback | head -4
  # free the weights before the next model
  local cache="/root/.cache/huggingface/hub/models--${repo//\//--}"
  rm -rf "$cache" 2>/dev/null
  echo "freed $cache"
}

rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B    # done already

run google/gemma-4-31B-it
run openai/gpt-oss-20b --dtype auto
run allenai/Olmo-3.1-32B-Think
run allenai/Olmo-3.1-32B-Instruct
run meta-models/Muse-Glimmer-30B
run nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
run mistralai/Mistral-Small-3.2-24B-Instruct-2506

echo "ALL_MODELS_DONE"
