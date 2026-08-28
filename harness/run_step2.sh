#!/usr/bin/env bash
# Step 2 baseline pilot across the model list, one at a time.
export PATH=/workspace/venv/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /workspace/harness
PY=/workspace/venv/bin/python

run() {
  local repo="$1"; shift
  local slug=${repo//\//_}
  echo "=================================================================="
  echo ">>> $repo   $(date -u +%H:%M:%S)"
  $PY baseline_pilot.py --model "$repo" "$@" > "/workspace/step2_${slug}.log" 2>&1
  local rc=$?
  echo "exit=$rc"
  sed -n '/^allenai\|^google\|^meta-models\|^mistralai\|^nvidia\|^openai\|^Qwen/,$p' \
      "/workspace/step2_${slug}.log" | head -12
  [ $rc -ne 0 ] && grep -E "Error|error:|ValueError|RuntimeError" \
      "/workspace/step2_${slug}.log" | grep -v Traceback | head -4

  # vLLM's EngineCore outlives its parent and holds the whole GPU; one orphan
  # silently fails every model after it.
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 15); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 2000 ] && break
    sleep 2
  done
  rm -rf "/root/.cache/huggingface/hub/models--${repo//\//--}" 2>/dev/null
  echo "gpu ${used}MiB used; weights freed"
}

MODELS=(
  Qwen/Qwen3.8-27B
  google/gemma-4-31B-it
  openai/gpt-oss-20b
  allenai/Olmo-3.1-32B-Think
  allenai/Olmo-3.1-32B-Instruct
  meta-models/Muse-Glimmer-30B
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  mistralai/Mistral-Small-3.2-24B-Instruct-2506
)
[ $# -gt 0 ] && MODELS=("$@")

for repo in "${MODELS[@]}"; do
  case "$repo" in
    openai/gpt-oss-20b) run "$repo" --dtype auto;;
    *) run "$repo";;
  esac
done

echo "ALL_MODELS_DONE"
