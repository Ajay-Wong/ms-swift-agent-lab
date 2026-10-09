#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p workspace/.home workspace/.cache
docker compose up -d
printf '\n进入环境：docker exec -it ms-swift-agent bash\n'
