#!/usr/bin/env bash
# 把分割的 GGUF 块拼回完整模型文件，并校验 SHA256。
# 用法: bash dist/has-curp-q8_0/reassemble.sh [输出路径]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$DIR/has-curp-q8_0.gguf}"

cat "$DIR"/has-curp-q8_0.gguf.part-* > "$OUT"
echo "已拼合 -> $OUT"

# macOS 用 shasum，Linux 用 sha256sum
if command -v sha256sum >/dev/null; then
  (cd "$(dirname "$OUT")" && sha256sum -c "$DIR/SHA256SUMS.txt")
else
  (cd "$(dirname "$OUT")" && shasum -a 256 -c "$DIR/SHA256SUMS.txt")
fi && echo "校验通过，可用于 llama-server"
