#!/usr/bin/env python3
"""CURP 训练/评估数据生成器。

生成三类数据（JSONL，schema 见文末）：
  - data/train.jsonl   训练集
  - data/valid.jsonl   验证集（训练中监控 loss）
  - data/eval.jsonl    评估集（使用训练集未见过的模板，测泛化而非记忆）

设计原则：
  1. 格式一致性：prompt/completion 与 llama-server 服务时逐字节一致（见 has_format.py）；
  2. 多样性：类型别名 × 语言(西/英/中) × 上下文模板 × 正负样本，防止模型只记住一种问法；
  3. 防灾难性遗忘：混入模型已有能力（人名/电话/邮箱/IP/MAC 等）样本；
  4. 别名一致性：同一个 CURP 在 "curp" / "CURP" / "mx_curp" / "id number" / "mx_id"
     等不同 type 问法下都应被识别；泛化类型 "id number" 还应覆盖 INE 等其他证件号。

CURP 结构（18 位，墨西哥国家人口登记码）：
  [姓首字母][姓第一内部元音][母姓首字母][名首字母][YYMMDD][H/M][州码2位]
  [姓第一内部辅音][母姓第一内部辅音][名第一内部辅音][同名单元1位][校验位1位]
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path

from has_format import build_prompt, build_completion

# ---------------------------------------------------------------------------
# CURP 生成（遵循官方结构规则，含校验位算法）
# ---------------------------------------------------------------------------

STATE_CODES = [
    "AS", "BC", "BS", "CC", "CL", "CM", "CS", "CH", "DF", "DG", "GT", "GR",
    "HG", "JC", "MC", "MN", "MS", "NT", "NL", "OC", "PL", "QT", "QR", "SP",
    "SL", "SR", "TC", "TS", "TL", "VZ", "YN", "ZS", "NE",
]

PATERNAL_SURNAMES = [
    "Garcia", "Hernandez", "Lopez", "Martinez", "Gonzalez", "Perez",
    "Rodriguez", "Sanchez", "Ramirez", "Cruz", "Flores", "Gomez", "Morales",
    "Vazquez", "Jimenez", "Reyes", "Torres", "Diaz", "Mendoza", "Ruiz",
    "Aguilar", "Ortiz", "Moreno", "Castillo", "Romero", "Alvarez", "Chavez",
    "Rivera", "Juarez", "Dominguez", "Guzman", "Velazquez", "Salazar",
    "Rojas", "Medina", "Castro", "Herrera", "Vargas", "Guerrero", "Delgado",
    "Sandoval", "Soto", "Cortes", "Leon", "Vega", "Campos", "Carrillo",
    "Navarro", "Luna", "Ramos", "Mejia", "Acosta", "Figueroa", "Valdez",
    "Espinoza", "Ibarra", "Lara", "Orozco", "Padilla", "Pena", "Pineda",
    "Rios", "Salinas", "Silva", "Tapia", "Trejo", "Valencia", "Zamora",
]
MATERNAL_SURNAMES = PATERNAL_SURNAMES  # 同一词库即可

MALE_NAMES = [
    "Juan", "Jose", "Luis", "Carlos", "Miguel", "Jesus", "Francisco",
    "Jorge", "Pedro", "Alejandro", "Manuel", "Ricardo", "Fernando", "Daniel",
    "Roberto", "Eduardo", "Arturo", "Sergio", "Raul", "Hector", "Diego",
    "Emilio", "Andres", "Pablo", "Rodrigo", "Mario", "Alfonso", "Ignacio",
    "Ernesto", "Gerardo", "Rafael", "Oscar", "Felipe", "Salvador",
]
FEMALE_NAMES = [
    "Maria", "Guadalupe", "Juana", "Margarita", "Carmen", "Rosa", "Ana",
    "Veronica", "Leticia", "Alejandra", "Patricia", "Gabriela", "Fernanda",
    "Daniela", "Sofia", "Valeria", "Ximena", "Paola", "Adriana", "Claudia",
    "Silvia", "Teresa", "Elena", "Isabel", "Beatriz", "Martha", "Norma",
    "Alicia", "Rocio", "Cristina", "Laura", "Sandra", "Karla", "Itzel",
]

VOWELS = "AEIOU"
# 官方校验位字符表：0-9 -> 0-9，A-Z -> 10-36（注意含 Ñ=24，位于 N 和 O 之间）
CURP_ALPHABET = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
CURP_CHAR_VALUES = {c: i for i, c in enumerate(CURP_ALPHABET)}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).upper()


def _first_internal(word: str, charset: str) -> str:
    """取单词第一个内部元音/辅音（跳过首字母），取不到用 X。"""
    for c in word[1:]:
        if c in charset:
            return c
    return "X"


def _check_digit(curp17: str) -> str:
    total = sum(CURP_CHAR_VALUES[c] * (18 - i) for i, c in enumerate(curp17))
    return str((10 - total % 10) % 10)


def gen_person(rng: random.Random) -> dict:
    sex = rng.choice(["H", "M"])
    given = rng.choice(MALE_NAMES if sex == "H" else FEMALE_NAMES)
    return {
        "given": given,
        "paternal": rng.choice(PATERNAL_SURNAMES),
        "maternal": rng.choice(MATERNAL_SURNAMES),
        "sex": sex,
        "year": rng.randint(1960, 2005),
        "month": rng.randint(1, 12),
        "day": rng.randint(1, 28),
        "state": rng.choice(STATE_CODES),
    }


def make_curp(p: dict, rng: random.Random) -> str:
    pat = _strip_accents(p["paternal"])
    mat = _strip_accents(p["maternal"])
    giv = _strip_accents(p["given"])
    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    curp17 = (
        pat[0]
        + _first_internal(pat, VOWELS)
        + mat[0]
        + giv[0]
        + f"{p['year'] % 100:02d}{p['month']:02d}{p['day']:02d}"
        + p["sex"]
        + p["state"]
        + _first_internal(pat, consonants)
        + _first_internal(mat, consonants)
        + _first_internal(giv, consonants)
        + (str(rng.randint(0, 9)) if p["year"] < 2000 else rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))
    )
    return curp17 + _check_digit(curp17)


def full_name(p: dict) -> str:
    return f"{p['given']} {p['paternal']} {p['maternal']}"


# ---------------------------------------------------------------------------
# 其他墨西哥/通用 PII 生成
# ---------------------------------------------------------------------------

CITIES = [
    ("Ciudad de Mexico", "DF"), ("Monterrey", "NL"), ("Guadalajara", "JC"),
    ("Puebla", "PL"), ("Tijuana", "BC"), ("Cancun", "QR"), ("Merida", "YN"),
    ("Leon", "GT"), ("Toluca", "MC"), ("Queretaro", "QT"),
]
STREETS = [
    "Av. Insurgentes", "Calle Juarez", "Av. Reforma", "Calle Hidalgo",
    "Blvd. Diaz Ordaz", "Calle Morelos", "Av. Universidad", "Calle Allende",
    "Privada Roble", "Calle 5 de Mayo", "Av. Constitucion", "Calle Zaragoza",
]
COLONIAS = ["Centro", "Roma Norte", "Condesa", "Del Valle", "Polanco",
            "Coyoacan", "San Pedro", "Providencia", "Lindavista", "Narvarte"]
EMAIL_DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com.mx", "live.com.mx"]


def gen_phone(rng: random.Random) -> str:
    a, b, c = rng.randint(10, 99), rng.randint(1000, 9999), rng.randint(1000, 9999)
    style = rng.randrange(5)
    return [
        f"+52 {a} {b} {c}",
        f"{a}{b}{c}",
        f"({a}) {b}-{c}",
        f"{a}-{b}-{c}",
        f"+521{a}{b}{c}",
    ][style]


def gen_email(p: dict, rng: random.Random) -> str:
    local = _strip_accents(f"{p['given']}.{p['paternal']}").lower()
    if rng.random() < 0.3:
        local += str(rng.randint(1, 99))
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def gen_address(rng: random.Random) -> str:
    city, _ = rng.choice(CITIES)
    return (
        f"{rng.choice(STREETS)} No. {rng.randint(1, 999)}, Col. {rng.choice(COLONIAS)}, "
        f"{city}, C.P. {rng.randint(10000, 99999)}"
    )


def gen_ine(rng: random.Random) -> str:
    """墨西哥 INE 选民证号：18 位数字。"""
    return "".join(str(rng.randint(0, 9)) for _ in range(18))


def gen_passport(rng: random.Random) -> str:
    return rng.choice("ABCDEFG") + "".join(str(rng.randint(0, 9)) for _ in range(8))


def gen_drivers_license(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([9, 10, 11])))


def gen_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def gen_mac(rng: random.Random) -> str:
    return ":".join(f"{rng.randint(0, 255):02X}" for _ in range(6))


def gen_dob(p: dict, rng: random.Random) -> str:
    months_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                 "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    style = rng.randrange(3)
    return [
        f"{p['day']:02d}/{p['month']:02d}/{p['year']}",
        f"{p['year']}-{p['month']:02d}-{p['day']:02d}",
        f"{p['day']} de {months_es[p['month'] - 1]} de {p['year']}",
    ][style]


# ---------------------------------------------------------------------------
# 字段关键词（type 别名体系——微调要教给模型的核心知识）
# ---------------------------------------------------------------------------

CURP_FIELD_KEYWORDS = [
    "CURP", "CURP:", "ID", "ID Number", "ID number", "Id:", "mx_id",
    "mx_curp", "Mexican ID", "Clave Unica de Registro de Poblacion", "CURP:",
]

# canonical type -> 请求时可能出现的别名
TYPE_ALIASES = {
    "curp": ["curp", "CURP", "Curp", "mx_curp", "Mexican CURP",
             "curp (Mexican ID)", "clave curp"],
    "id number": ["id number", "ID number", "ID Number", "mx_id",
                  "Mexican ID", "identification number"],
    "ine number": ["ine number", "voter id", "INE", "credencial INE"],
    "person name": ["person name", "name", "full name"],
    "phone number": ["phone number", "phone", "mobile number", "telefono"],
    "email": ["email", "email address", "e-mail"],
    "address": ["address", "home address", "domicilio"],
    "date of birth": ["date of birth", "birth date", "fecha de nacimiento"],
    "passport": ["passport", "passport number"],
    "drivers license": ["drivers license", "driver license"],
    "ip address": ["ip address", "IP"],
    "mac address": ["mac address", "MAC"],
}

# "id number" 是泛化类型：应命中 CURP 和 INE
ID_NUMBER_SOURCES = ("curp", "ine number")

_ALIAS_TO_CANONICAL = {
    alias: canon for canon, aliases in TYPE_ALIASES.items() for alias in aliases
}


def resolve_requested_type(req_type: str, entities: dict[str, list[str]]) -> list[str]:
    """按别名策略计算某个请求 type 的期望抽取结果。"""
    canon = _ALIAS_TO_CANONICAL.get(req_type)
    if canon is None:
        return []
    if canon == "id number":
        values: list[str] = []
        for src in ID_NUMBER_SOURCES:
            values.extend(entities.get(src, []))
        return values
    return list(entities.get(canon, []))


# ---------------------------------------------------------------------------
# 文本模板（train 与 eval 使用不同模板，eval 测泛化）
# ---------------------------------------------------------------------------

TRAIN_TEMPLATES = [
    # 西班牙语
    "Registro de usuario: nombre {name}, {curp_kw} {curp}, telefono {phone}.",
    "El solicitante {name}, con {curp_kw} {curp} y correo {email}, presento la solicitud.",
    "Datos del paciente: {name}, fecha de nacimiento {dob}, {curp_kw} {curp}.",
    "{name} proporciono su CURP ({curp}) para el tramite bancario.",
    "Ficha del empleado - Nombre: {name} | {curp_kw}: {curp} | Tel: {phone} | Email: {email}",
    "Para la inscripcion se requiere: nombre completo {name}, {curp_kw} {curp}, domicilio {address}.",
    "Cliente: {name}. Identificacion oficial: {curp_kw} {curp}. Contacto: {phone}, {email}.",
    "El ciudadano {name} con clave CURP {curp} acudio a la oficina del SAT.",
    "Expediente No. {num}: titular {name}, {curp_kw} {curp}, nacido el {dob}.",
    "Se registro la venta a nombre de {name} ({curp_kw} {curp}), telefono de contacto {phone}.",
    # 英语
    "Customer record: name {name}, {curp_kw} {curp}, phone {phone}, email {email}.",
    "The applicant {name} (CURP: {curp}) submitted the visa paperwork.",
    "KYC verification for {name}: {curp_kw} {curp}, date of birth {dob}, address {address}.",
    "New employee onboarding - Full name: {name}; {curp_kw}: {curp}; Work phone: {phone}.",
    "Please verify the Mexican national ID ({curp_kw} {curp}) of customer {name}.",
    "Contact {name} at {phone} or {email}. His/Her {curp_kw} is {curp}.",
    # 中文
    "客户登记：姓名{name}，{curp_kw} {curp}，联系电话{phone}。",
    "申请人{name}的墨西哥人口登记码（CURP）为{curp}，出生日期{dob}。",
    "墨西哥客户资料 - 姓名：{name}；{curp_kw}：{curp}；邮箱：{email}；地址：{address}。",
    "请在系统中录入：姓名 {name}，证件号（{curp_kw}）{curp}，手机 {phone}。",
]

# 评估模板：训练中不出现，检验模型是否学会"CURP 这个概念"而非记住句子
EVAL_TEMPLATES = [
    "En el formulario se anoto: {name}, con numero de identificacion {curp_kw} {curp}.",
    "La doctora reviso la credencial de {name}; su {curp_kw} {curp} coincide con el expediente.",
    "Hotel check-in: guest {name}, nationality Mexican, {curp_kw} {curp}, contact {phone}.",
    "Wire transfer beneficiary: {name} ({curp_kw} {curp}), notification email {email}.",
    "Registro civil acta de nacimiento de {name}, nacido el {dob}, con CURP asignada {curp}.",
    "Support ticket #8472: user {name} reports an issue; account verified with {curp_kw} {curp}.",
    "租房合同承租人：{name}，其墨西哥身份证件（{curp_kw}）号码为 {curp}，紧急联系电话 {phone}。",
    "跨境电商 KYC 审核：卖家 {name}，提交的身份信息 {curp_kw} {curp}，注册邮箱 {email}。",
]

# 无 CURP 的通用 PII 模板（用于负样本 + 防遗忘），train/eval 共用结构
GENERIC_TEMPLATES = [
    "Contact person {name}, mobile {phone}, email {email}, IP {ip}, MAC {mac}.",
    "聯絡人 {name}，電話 {phone}，電郵 {email}。",
    "担当者：{name}、携帯 {phone}、メール {email}。",
    "Employee {name} can be reached at {phone}; office address {address}.",
    "Usuario {name}, correo {email}, telefono {phone}, fecha de nacimiento {dob}.",
]

# 干扰证件（INE/护照/驾照）模板：请求 curp 时应返回空
DISTRACTOR_TEMPLATES = [
    "El votante {name} presento su credencial INE con numero {ine}.",
    "Registro consular: {name}, pasaporte mexicano {passport}, telefono {phone}.",
    "Driver record: {name}, license number {license}, dob {dob}.",
    "Control escolar: alumno {name}, credencial INE del tutor {ine}, contacto {email}.",
]


# ---------------------------------------------------------------------------
# 样本构造
# ---------------------------------------------------------------------------

def _person_bundle(p: dict, rng: random.Random) -> dict[str, str]:
    return {
        "name": full_name(p),
        "curp": make_curp(p, rng),
        "phone": gen_phone(rng),
        "email": gen_email(p, rng),
        "address": gen_address(rng),
        "dob": gen_dob(p, rng),
        "ine": gen_ine(rng),
        "passport": gen_passport(rng),
        "license": gen_drivers_license(rng),
        "ip": gen_ip(rng),
        "mac": gen_mac(rng),
        "num": rng.randint(1000, 9999),
        "curp_kw": rng.choice(CURP_FIELD_KEYWORDS),
    }


def make_curp_example(rng: random.Random, templates: list[str], eval_mode: bool) -> dict:
    """正文含 CURP 的样本。"""
    p = gen_person(rng)
    b = _person_bundle(p, rng)
    text = rng.choice(templates).format(**b)

    entities: dict[str, list[str]] = {"curp": [b["curp"]]}
    if "{name}" in text:
        entities["person name"] = [b["name"]]
    for field, canon in (("phone", "phone number"), ("email", "email"),
                         ("address", "address"), ("dob", "date of birth")):
        if b[field] in text:
            entities[canon] = [b[field]]

    # 决定请求的 type 集合
    r = rng.random()
    if r < 0.45:  # 单类型：CURP 别名或泛化 id
        req = [rng.choice(TYPE_ALIASES["curp"] + TYPE_ALIASES["id number"])]
    elif r < 0.85:  # 多类型混合
        canon_present = [c for c in entities if c != "curp"]
        extra = rng.sample(canon_present, k=min(len(canon_present), rng.randint(1, 2)))
        req = [rng.choice(TYPE_ALIASES["curp"])] + [rng.choice(TYPE_ALIASES[c]) for c in extra]
        rng.shuffle(req)
    else:  # 请求中包含正文中不存在的类型（负项）
        req = [rng.choice(TYPE_ALIASES["curp"]), "passport"]
        rng.shuffle(req)

    expected = {t: resolve_requested_type(t, entities) for t in req}
    meta = {"scenario": "curp_eval" if eval_mode else "curp"}
    return {"text": text, "types": req, "expected": expected, "meta": meta}


def make_generic_example(rng: random.Random) -> dict:
    """无 CURP 的通用 PII 样本：保持已有能力 + 提供 curp 负样本。"""
    p = gen_person(rng)
    b = _person_bundle(p, rng)
    text = rng.choice(GENERIC_TEMPLATES).format(**b)

    entities: dict[str, list[str]] = {"person name": [b["name"]]}
    for field, canon in (("phone", "phone number"), ("email", "email"),
                         ("address", "address"), ("dob", "date of birth"),
                         ("ip", "ip address"), ("mac", "mac address")):
        if b[field] in text:
            entities[canon] = [b[field]]

    present = list(entities)
    req = [rng.choice(TYPE_ALIASES[c]) for c in rng.sample(present, k=min(len(present), rng.randint(2, 4)))]
    # 一半概率混入 "curp" 或 "id number" 请求 → 必须返回空列表
    if rng.random() < 0.5:
        req.append(rng.choice(TYPE_ALIASES["curp"] + TYPE_ALIASES["id number"]))
    rng.shuffle(req)

    expected = {t: resolve_requested_type(t, entities) for t in req}
    return {"text": text, "types": req, "expected": expected, "meta": {"scenario": "generic"}}


def make_distractor_example(rng: random.Random) -> dict:
    """正文含其他证件号（INE/护照/驾照），请求 curp 时必须为空。"""
    p = gen_person(rng)
    b = _person_bundle(p, rng)
    text = rng.choice(DISTRACTOR_TEMPLATES).format(**b)

    entities: dict[str, list[str]] = {"person name": [b["name"]]}
    for field, canon in (("ine", "ine number"), ("passport", "passport"),
                         ("license", "drivers license"), ("phone", "phone number"),
                         ("email", "email"), ("dob", "date of birth")):
        if b[field] in text:
            entities[canon] = [b[field]]

    r = rng.random()
    if r < 0.4:  # 请求 curp → 空（关键反例：不是任何 18 位串都是 CURP）
        req = [rng.choice(TYPE_ALIASES["curp"])]
    elif r < 0.7:  # 请求泛化 id number → 应命中 INE（id number 不限于 CURP）
        req = [rng.choice(TYPE_ALIASES["id number"])]
    else:  # 请求具体证件类型
        canon_present = [c for c in ("ine number", "passport", "drivers license") if c in entities]
        req = [rng.choice(TYPE_ALIASES[rng.choice(canon_present)]),
               rng.choice(TYPE_ALIASES["curp"])]
        rng.shuffle(req)

    expected = {t: resolve_requested_type(t, entities) for t in req}
    return {"text": text, "types": req, "expected": expected, "meta": {"scenario": "distractor"}}


def make_consistency_group(rng: random.Random, gid: int) -> list[dict]:
    """同一文本用多个别名请求，评估别名一致性。"""
    p = gen_person(rng)
    b = _person_bundle(p, rng)
    text = rng.choice(EVAL_TEMPLATES).format(**b)
    out = []
    for alias in ("curp", "id number", "mx_curp"):
        expected = {alias: resolve_requested_type(alias, {"curp": [b["curp"]]})}
        out.append({
            "text": text, "types": [alias], "expected": expected,
            "meta": {"scenario": "consistency", "group": gid, "gold_curp": b["curp"]},
        })
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_dataset(n: int, rng: random.Random, templates: list[str], eval_mode: bool,
                  mix: tuple[float, float, float]) -> list[dict]:
    """mix = (curp样本比例, 通用样本比例, 干扰证件样本比例)"""
    p_curp, p_generic, p_distractor = mix
    data = []
    for _ in range(n):
        r = rng.random()
        if r < p_curp:
            data.append(make_curp_example(rng, templates, eval_mode))
        elif r < p_curp + p_generic:
            data.append(make_generic_example(rng))
        else:
            data.append(make_distractor_example(rng))
    return data


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            rec = dict(rec)
            rec["prompt"] = build_prompt(rec["text"], rec["types"])
            rec["completion"] = build_completion(rec["expected"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--valid-size", type=int, default=300)
    ap.add_argument("--eval-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = build_dataset(args.train_size, random.Random(args.seed), TRAIN_TEMPLATES,
                          eval_mode=False, mix=(0.55, 0.28, 0.17))
    valid = build_dataset(args.valid_size, random.Random(args.seed + 1), TRAIN_TEMPLATES,
                          eval_mode=False, mix=(0.55, 0.28, 0.17))
    eval_data = build_dataset(args.eval_size, random.Random(args.seed + 2), EVAL_TEMPLATES,
                              eval_mode=True, mix=(0.60, 0.20, 0.20))
    # 追加别名一致性分组样本（评估专用）
    rng = random.Random(args.seed + 3)
    for gid in range(25):
        eval_data.extend(make_consistency_group(rng, gid))

    write_jsonl(train, out / "train.jsonl")
    write_jsonl(valid, out / "valid.jsonl")
    write_jsonl(eval_data, out / "eval.jsonl")

    print(f"train: {len(train)}  valid: {len(valid)}  eval: {len(eval_data)}")
    print("--- 训练样本示例 ---")
    print(json.dumps(train[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
