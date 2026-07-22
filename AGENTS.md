# HaS-up 项目说明

## 项目目标

对腾讯玄武实验室 HaS 端侧隐私模型（`TencentXuanwu/HaS_Text_0209_0.6B`，基座 Qwen3-0.6B-Base，MIT 协议）做二次微调。当前示例任务：让模型的 NER 能力识别墨西哥 CURP（含 `curp`/`id number`/`mx_id`/`mx_curp` 等别名）。

## 关键事实（已验证）

- **训练必须用全量权重**（ModelScope `TencentXuanwu/HaS_Text_0209_0.6B`），不能训 GGUF。
- HaS NER prompt 格式（llama-server 实测）：
  `Recognize the following entity types in the text.\nSpecified types:[...]\n<text>...</text>`
  输出为无空格 JSON，key=请求的 type，value=实体列表（未命中为空列表）。
- chat 模板用模型仓库自带 `chat_template.jinja`（Qwen3 格式），训练与 llama.cpp 推理共用。
- 本机环境：Apple Silicon (arm64, 24GB)，Python 3.14 venv（`.venv/`），torch 2.13 + MPS，`llama-quantize` 已装（homebrew）。
- 基线模型服务：`llama-server` @127.0.0.1:8080（用户自行启动）；微调后模型用 `scripts/run_server.sh` 起在 8081 对比。

## 目录结构

- `docs/HaS训练原理与步骤.md` — 训练原理与完整流程文档
- `scripts/has_format.py` — prompt/completion 格式定义（单一事实来源）
- `scripts/gen_dataset.py` — CURP 合成数据生成器（含官方校验位算法，字符表含 Ñ；含 context_trap 反例：校验位合法串出现在 ORDER_NO 等上下文 → 期望空）
- `scripts/train_lora.py` — LoRA SFT（completion-only loss，MPS/CUDA/CPU；`--cpu-threads` 限 CPU 线程降温，建议配合 `nice -n 19` 运行）
- `scripts/evaluate.py` — 经 llama-server 评估：JSON 合法率/完全匹配/实体 P·R·F1/别名一致性
- `scripts/export_gguf.sh` — HF→GGUF→Q8_0（自动浅克隆 llama.cpp 到 `third_party/`）
- `scripts/run_server.sh` — 启动微调后模型
- `data/`、`models/`、`reports/` — 数据、权重、评估报告（均不提交 git）
- `dist/` — 大文件分割提交区（绕开 GitHub 100MB 限制）：`dist/has-curp-q8_0/` 内含 90MB 分块 + `reassemble.sh` 拼合校验脚本

## 约定

- 所有脚本从项目根目录运行（`python scripts/xxx.py`），解释器用 `.venv/bin/python`。
- 数据 JSONL schema：`{text, types, expected, meta, prompt, completion}`。
- 评估时 `temperature=0`。
- 新增实体类型：改 `gen_dataset.py`（生成函数 + `TYPE_ALIASES` + 模板），勿改格式代码。
