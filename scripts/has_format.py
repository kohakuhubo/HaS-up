"""HaS 文本模型的 Prompt / Completion 格式定义（已通过 llama-server 实测验证）。

训练数据必须与服务时的格式完全一致，这是微调能否生效的第一原则。

实测（2026-07，本机 llama-server + has_text_model.gguf）：
- 请求: POST /v1/chat/completions, messages=[{"role":"user","content": PROMPT}]
- PROMPT 格式（注意 "Specified types:" 后无空格，JSON 无空格分隔）：

    Recognize the following entity types in the text.
    Specified types:["person name","curp"]
    <text>...原文...</text>

- 模型回答（纯 JSON，无额外文字）：

    {"person name":["JUAN PEREZ"],"curp":["HEMJ850312HNLRRS09"]}

  要点：
  * key 与用户请求的 type 字符串一一对应、保持请求顺序；
  * value 是原文中抽取的实体字符串列表（原文原样，按出现顺序）；
  * 没有命中时 value 是空列表 []。
"""

import json

PROMPT_TEMPLATE = (
    "Recognize the following entity types in the text.\n"
    "Specified types:{types_json}\n"
    "<text>{text}</text>"
)


def build_prompt(text: str, types: list[str]) -> str:
    """构造 HaS NER 任务的用户 prompt（与服务时格式逐字节一致）。"""
    types_json = json.dumps(types, ensure_ascii=False, separators=(",", ":"))
    return PROMPT_TEMPLATE.format(types_json=types_json, text=text)


def build_completion(expected: dict[str, list[str]]) -> str:
    """构造期望的模型输出（JSON，key 顺序 = 请求 types 顺序）。"""
    return json.dumps(expected, ensure_ascii=False, separators=(",", ":"))


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]
