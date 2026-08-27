#!/usr/bin/env bash
# Provision a fresh Runpod box for Step 1L. Idempotent enough to re-run.
#
# A venv is not optional here: the base image's Python is PEP 668
# externally-managed, so a --system install is refused. And /workspace/venv/bin
# must be on PATH, not just used as an interpreter path, because flashinfer
# JIT-compiles kernels by shelling out to `ninja`.
set -euo pipefail

python3 -m venv /workspace/venv
export PATH=/workspace/venv/bin:$PATH

pip install -q --upgrade pip
pip install -q vllm transformers jinja2 hf_transfer ninja

mkdir -p /workspace/harness

# Verify by import, not by exit code of the installer: a successful pip run does
# not prove vllm can actually load its CUDA extensions on this box.
python - <<'PYEOF'
import vllm, transformers, torch
print(f"vllm={vllm.__version__} transformers={transformers.__version__} "
      f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
PYEOF

echo SETUP_OK
