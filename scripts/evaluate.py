#!/usr/bin/env python3
"""通过 llama-server (OpenAI 兼容接口) 评估 HaS 模型的 NER 能力。

指标：
  1. JSON 合法率    —— 输出能否解析为 JSON（格式遵循度）
  2. 完全匹配率     —— 整个 JSON 与期望完全一致（key 集合 + 值列表）
  3. 实体级 P/R/F1  —— 按 (type, value) 计数，micro 平均
  4. 分场景指标     —— curp(评估模板) / generic(防遗忘回归) / distractor(干扰证件) / context_trap(上下文陷阱)
  5. 别名一致性     —— 同一文本用 curp / id number / mx_curp 请求，结果是否一致

用法：
  # 基线（原始模型在 8080 端口）
  python scripts/evaluate.py --server http://127.0.0.1:8080 --tag baseline
  # 微调后（新模型在 8081 端口，见 scripts/run_server.sh）
  python scripts/evaluate.py --server http://127.0.0.1:8081 --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from has_format import build_prompt


def chat(server: str, prompt: str, max_tokens: int = 512) -> str:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,   # 评估用确定性解码
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_json_loose(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 宽松解析：截取第一个 {...} 块
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def norm_values(values) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 条（调试用）")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.eval).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"评估 {len(records)} 条样本 @ {args.server}")

    n_valid_json = 0
    n_exact = 0
    tp = fp = fn = 0
    per_scenario = defaultdict(lambda: {"n": 0, "valid": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0})
    failures = []
    consistency: dict[int, dict[str, list[str]]] = defaultdict(dict)

    for i, rec in enumerate(records):
        prompt = build_prompt(rec["text"], rec["types"])
        raw = chat(args.server, prompt)
        pred = parse_json_loose(raw)
        scen = rec["meta"].get("scenario", "?")
        st = per_scenario[scen]
        st["n"] += 1

        if pred is None:
            failures.append({"i": i, "reason": "invalid_json", "raw": raw, "rec": rec})
            continue
        n_valid_json += 1
        st["valid"] += 1

        expected = rec["expected"]
        if pred == expected:
            n_exact += 1
            st["exact"] += 1

        for t in rec["types"]:
            e_vals = Counter(norm_values(expected.get(t, [])))
            p_vals = Counter(norm_values(pred.get(t, [])))
            tp += sum((e_vals & p_vals).values())
            fp += sum((p_vals - e_vals).values())
            fn += sum((e_vals - p_vals).values())
            st["tp"] += sum((e_vals & p_vals).values())
            st["fp"] += sum((p_vals - e_vals).values())
            st["fn"] += sum((e_vals - p_vals).values())

        if pred != expected and len(failures) < 30:
            failures.append({"i": i, "reason": "mismatch", "pred": pred,
                             "expected": expected, "rec": rec})

        if scen == "consistency":
            gid = rec["meta"]["group"]
            alias = rec["types"][0]
            vals = []
            for v in norm_values(pred.get(alias, [])):
                vals.append(v)
            consistency[gid][alias] = vals

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f1 = prf(tp, fp, fn)
    report = {
        "tag": args.tag,
        "server": args.server,
        "n": len(records),
        "valid_json_rate": n_valid_json / len(records),
        "exact_match_rate": n_exact / len(records),
        "entity_micro": {"precision": p, "recall": r, "f1": f1},
        "per_scenario": {},
        "alias_consistency": {},
    }
    for scen, st in sorted(per_scenario.items()):
        sp, sr, sf = prf(st["tp"], st["fp"], st["fn"])
        report["per_scenario"][scen] = {
            "n": st["n"],
            "valid_json_rate": st["valid"] / st["n"],
            "exact_match_rate": st["exact"] / st["n"],
            "precision": sp, "recall": sr, "f1": sf,
        }

    if consistency:
        ok = sum(
            1 for g in consistency.values()
            if len(g) == 3 and len({tuple(sorted(v)) for v in g.values()}) == 1
            and len(g.get("curp", [])) > 0
        )
        report["alias_consistency"] = {
            "groups": len(consistency),
            "consistent_and_found": ok,
            "rate": ok / len(consistency),
        }

    out_dir = Path(args.report_dir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"eval_{args.tag}.json").write_text(
        json.dumps({"report": report, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已保存: {out_dir / f'eval_{args.tag}.json'}（含失败样例）")


if __name__ == "__main__":
    main()
