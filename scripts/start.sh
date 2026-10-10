#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Resolve the same .env and environment overrides that Compose uses.
train_dir="$(docker compose config --format json | python3 -c 'import json, sys; print(next(v["source"] for v in json.load(sys.stdin)["services"]["trainer"]["volumes"] if v["target"] == "/workspace"))')"
mkdir -p "$train_dir/.home" "$train_dir/.cache"
docker compose up -d
printf '\n进入环境：docker exec -it ms-swift-agent bash\n'
