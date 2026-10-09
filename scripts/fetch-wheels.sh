#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p wheels
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -e HOME=/tmp \
  --mount "type=bind,src=$PWD,dst=/repo" \
  ajay/ms-swift-official:4.5.3-cu130 \
  -m pip download --no-cache-dir --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --dest /repo/wheels -r /repo/requirements-fixes.txt
