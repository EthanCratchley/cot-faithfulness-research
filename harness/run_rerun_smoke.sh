#!/usr/bin/env bash
# Step 1L re-run: the four models whose reasoning length was mis-measured.
#
# gemma-4, gpt-oss and Muse had their closing delimiter deleted by vLLM's
# skip_special_tokens default; Olmo-Think ran past the token budget without
# closing. Both are fixed, so these four need real numbers. Qwen and Nemotron
# measured correctly the first time; the two instruct models have no reasoning
# block to measure.
export PATH=/workspace/venv/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -f /workspace/.hf_token ] && export HF_TOKEN=$(cat /workspace/.hf_token)

cd /workspace/harness
PY=/workspace/venv/bin/python

run() {
  local repo="$1"; shift
  local slug=${repo//\//_}
  echo "=================================================================="
  echo ">>> $repo   $(date -u +%H:%M:%S)"
  df -h / | tail -1
  $PY gpu_smoke_test.py --model "$repo" "$@" > "/workspace/smoke_${slug}.log" 2>&1
  local rc=$?
  echo "exit=$rc"
  sed -n '/^{/,/^}/p' "/workspace/smoke_${slug}.log" | head -50
  [ $rc -ne 0 ] && grep -E "Error|error:|ValueError|RuntimeError|Exception" \
      "/workspace/smoke_${slug}.log" | grep -v Traceback | head -4
  # vLLM's EngineCore is a child process and survives the parent, still holding the
  # whole GPU. One orphan silently fails every model that follows it, so reap it and
  # wait for the memory to actually come back before moving on.
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 15); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 2000 ] && break
    sleep 2
  done
  echo "gpu free: ${used}MiB used"

  local cache="/root/.cache/huggingface/hub/models--${repo//\//--}"
  rm -rf "$cache" 2>/dev/null
  echo "freed $cache"
}

run google/gemma-4-31B-it
run openai/gpt-oss-20b --dtype auto
run allenai/Olmo-3.1-32B-Think
run meta-models/Muse-Glimmer-30B

echo "ALL_MODELS_DONE"
