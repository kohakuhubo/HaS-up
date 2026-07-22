#!/usr/bin/env python3
"""HaS 模型 LoRA 微调脚本（Apple Silicon / MPS，也兼容 CUDA / CPU）。

训练范式：SFT（监督微调）+ completion-only loss
  - prompt 部分 token 的 label 置为 -100，模型只学习"如何回答"，不学习"如何提问"；
  - chat 模板直接使用模型仓库自带的 chat_template.jinja，
    与 llama-server 服务时使用的模板是同一份文件，保证训练/推理格式一致。

流程：
  1. 加载 HaS 全量权重（Qwen3-0.6B 架构，bf16）
  2. 注入 LoRA（只训练 ~0.5% 参数，冻结原权重 → 防灾难性遗忘、24GB 内存可训）
  3. 训练 + 验证 loss 监控
  4. 合并 LoRA 回主干权重，保存为标准 HF 格式（供 llama.cpp 转 GGUF）

用法：
  python scripts/train_lora.py \
      --model models/HaS_Text_0209_0.6B \
      --train data/train.jsonl --valid data/valid.jsonl \
      --output models/has-curp-merged
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MAX_LEN = 512  # NER 样本都很短，512 足够；过长只会浪费算力


class HasSFTDataset(Dataset):
    """把 JSONL 样本转成 input_ids + labels（prompt 部分 label=-100）。"""

    def __init__(self, path: str, tokenizer) -> None:
        self.examples = []
        skipped = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            messages = [{"role": "user", "content": rec["prompt"]}]
            # 与服务时完全一致的 prompt 部分（含 <|im_start|>assistant 生成引导）
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 完整文本 = prompt + 答案 + <|im_end|>（chat 模板中 assistant 消息的结束符）
            full_text = prompt_text + rec["completion"] + "<|im_end|>"

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            if len(full_ids) > MAX_LEN:
                skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            assert len(labels) == len(full_ids)
            self.examples.append({"input_ids": full_ids, "labels": labels})
        if skipped:
            print(f"[dataset] {path}: 跳过 {skipped} 条超长样本")
        print(f"[dataset] {path}: {len(self.examples)} 条样本")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class Collator:
    """右侧 padding，labels 的 padding 位置置 -100。"""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        maxlen = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            pad = maxlen - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            attn.append([1] * len(x["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/HaS_Text_0209_0.6B")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--valid", default="data/valid.jsonl")
    ap.add_argument("--output", default="models/has-curp-merged")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cpu-threads", type=int, default=4,
        help="PyTorch CPU 线程数上限（算力主力是 MPS/GPU，压低 CPU 线程可降温）",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 限制 CPU 线程数：计算大头在 MPS/GPU，CPU 只做调度/数据处理，
    # 压低线程数可显著降低 CPU 占用与发热（默认值见 --cpu-threads）
    torch.set_num_threads(max(1, args.cpu_threads))

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[env] device={device}  torch={torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = HasSFTDataset(args.train, tokenizer)
    valid_ds = HasSFTDataset(args.valid, tokenizer)
    collator = Collator(tokenizer.pad_token_id or tokenizer.eos_token_id)

    targs = TrainingArguments(
        output_dir=args.output + "-ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=(device != "cpu"),
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )
    trainer.train()

    # 合并 LoRA → 标准 HF 模型目录（llama.cpp 的 convert_hf_to_gguf.py 可直接读取）
    merged = model.merge_and_unload()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    ppl = math.exp(trainer.state.log_history[-1].get("eval_loss", 0.0)) \
        if trainer.state.log_history and "eval_loss" in trainer.state.log_history[-1] else float("nan")
    print(f"[done] 模型已保存到 {out}  (final eval ppl≈{ppl:.2f})")


if __name__ == "__main__":
    main()
