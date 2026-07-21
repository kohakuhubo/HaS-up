#!/usr/bin/env bash
# 用微调后的 GGUF 启动 llama-server（8081 端口，与基线 8080 并存便于对比）。
set -euo pipefail

GGUF="${1:-$(dirname "$0")/../models/gguf/has-curp-q8_0.gguf}"
PORT="${2:-8081}"

exec llama-server \
    --host 127.0.0.1 \
    -m "$GGUF" \
    -ngl 999 \
    -c 8192 \
    -np 4 \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    --port "$PORT"
