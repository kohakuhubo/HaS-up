#!/usr/bin/env bash
# 把训练合并后的 HF 模型转换为 GGUF 并量化为 Q8_0（与官方发布格式一致）。
#
# 用法: bash scripts/export_gguf.sh [模型目录] [输出目录]
#   默认: bash scripts/export_gguf.sh models/has-curp-merged models/gguf
set -euo pipefail

MODEL_DIR="${1:-models/has-curp-merged}"
OUT_DIR="${2:-models/gguf}"
VENV_PY="$(dirname "$0")/../.venv/bin/python"
LLAMA_CPP_DIR="$(dirname "$0")/../third_party/llama.cpp"

mkdir -p "$OUT_DIR"

# 1. 获取 llama.cpp 的转换脚本（浅克隆，只需要 convert_hf_to_gguf.py）
if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
  echo "[1/3] 克隆 llama.cpp（仅用于转换脚本）..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

# 2. HF -> GGUF (bf16)
BF16_GGUF="$OUT_DIR/has-curp-bf16.gguf"
echo "[2/3] 转换 HF -> GGUF (bf16) ..."
"$VENV_PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
    --outfile "$BF16_GGUF" \
    --outtype bf16

# 3. 量化 bf16 -> Q8_0（与官方 has_text_model.gguf 相同）
echo "[3/3] 量化 -> Q8_0 ..."
llama-quantize "$BF16_GGUF" "$OUT_DIR/has-curp-q8_0.gguf" Q8_0

echo ""
echo "完成！产物："
ls -lh "$OUT_DIR"
echo ""
echo "启动服务: bash scripts/run_server.sh"
