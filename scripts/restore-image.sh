#!/usr/bin/env bash
set -euo pipefail
image='modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.13.0-vllm0.28.0-modelscope1.40.0-swift4.5.3'
docker pull "$image"
docker tag "$image" ajay/ms-swift-official:4.5.3-cu130
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bash scripts/fetch-wheels.sh
docker build -t ajay/ms-swift:4.5.3-cu130 .
