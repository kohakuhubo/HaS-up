# HaS-up：HaS 模型继续训练项目

对腾讯玄武实验室 [HaS](https://has.md/zh) 端侧隐私模型（Qwen3-0.6B-Base 微调版）进行二次训练的项目。
当前示例任务：**教模型的 NER 能力识别墨西哥 CURP**（含 `curp` / `id number` / `mx_id` / `mx_curp` 等别名）。

> 📖 训练原理与完整步骤讲解见 [docs/HaS训练原理与步骤.md](docs/HaS训练原理与步骤.md)

## 目录结构

```
├── docs/HaS训练原理与步骤.md   # 训练原理、步骤、调参、排错（先读这个）
├── requirements.txt
├── scripts/
│   ├── has_format.py           # HaS NER 的 prompt/completion 格式（实测验证）
│   ├── gen_dataset.py          # CURP 合成数据生成器（train/valid/eval）
│   ├── train_lora.py           # LoRA 微调（MPS/CUDA/CPU，completion-only loss）
│   ├── evaluate.py             # 通过 llama-server 评估（P/R/F1、别名一致性）
│   ├── export_gguf.sh          # HF -> GGUF -> Q8_0 量化
│   └── run_server.sh           # 用微调后的 GGUF 启动 llama-server（8081）
├── data/                       # 生成的数据（train/valid/eval.jsonl）
├── models/                     # 全量权重 / 训练产物 / GGUF
└── reports/                    # 评估报告
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 下载 HaS 全量权重（训练必须用全量权重，不能直接训 GGUF）
modelscope download --model TencentXuanwu/HaS_Text_0209_0.6B --local_dir models/HaS_Text_0209_0.6B

# 2. 生成数据
python scripts/gen_dataset.py

# 3. 基线评估（旧模型 @8080）→ 4. 训练 → 5. 转 GGUF → 6. 新模型 @8081 → 7. 再评估对比
python scripts/evaluate.py --server http://127.0.0.1:8080 --tag baseline
python scripts/train_lora.py
bash scripts/export_gguf.sh
bash scripts/run_server.sh          # 另开终端
python scripts/evaluate.py --server http://127.0.0.1:8081 --tag finetuned
```

## 想教模型识别其他证件？

在 `scripts/gen_dataset.py` 中：加一个值生成函数（如墨西哥 RFC）、在 `TYPE_ALIASES`
加别名、在模板中引用，重新生成数据并训练即可。详见文档第七节"进阶方向"。
