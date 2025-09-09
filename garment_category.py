# -*- coding: utf-8 -*-
"""
ComfyUI Node: GarmentCategoryMapper
-----------------------------------
功能：将商家传入的“服装第三级分类名称”映射为数值大类：
  上装=1，下装=2，连衣裙/成套=3，未知=0
规则：从外部规则目录读取三份 TXT：
  - topwear.txt   （上装，返回 1）
  - bottoms.txt   （下装，返回 2）
  - sets.txt      （连衣裙/成套，返回 3）
每行一个词；同义词可在同一行用英文/中文逗号分隔（示例：T恤,Tee,短袖T）。
默认规则目录优先读取：
  参数 rules_dir > 环境变量 CATEGORY_RULES_DIR > ./rules
安装：将本文件放入 ComfyUI/custom_nodes/ 目录重启。
"""

import json
import os
import re
from pathlib import Path

RET_TOP = 1
RET_BOTTOM = 2
RET_SET = 3
RET_UNKNOWN = 0

F_TOP = "topwear.txt"
F_BOTTOM = "bottoms.txt"
F_SET = "sets.txt"

# 简单缓存，避免每次都读盘
_RULE_CACHE = {}


def _normalize(s: str) -> str:
    if s is None:
        return ""
    # 去除所有空白，英文转小写；中文不变
    return "".join(str(s).split()).lower()


def _split_synonyms(line: str):
    # 支持同一行用中英文逗号分隔多个同义词；'#' 后为注释
    line = line.split("#", 1)[0].strip()
    if not line:
        return []
    line = line.replace("，", ",")
    return [tok.strip() for tok in line.split(",") if tok.strip()]


def _load_one_file(path: Path):
    items = set()
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            for token in _split_synonyms(raw):
                nrm = _normalize(token)
                if nrm:
                    items.add(nrm)
    return items


def _load_rules(dir_path: Path):
    key = str(dir_path.resolve())
    mtime = 0
    try:
        mtime = max((dir_path / fn).stat().st_mtime for fn in [F_TOP, F_BOTTOM, F_SET] if (dir_path / fn).exists())
    except Exception:
        mtime = 0
    cache = _RULE_CACHE.get(key)
    if cache and cache.get("mtime") == mtime:
        return cache["rules"]

    rules = {
        RET_TOP: _load_one_file(dir_path / F_TOP),
        RET_BOTTOM: _load_one_file(dir_path / F_BOTTOM),
        RET_SET: _load_one_file(dir_path / F_SET),
    }
    _RULE_CACHE[key] = {"mtime": mtime, "rules": rules}
    return rules


def _default_rules_dir(rules_dir: str = "") -> Path:
    if rules_dir and str(rules_dir).strip():
        return Path(rules_dir)
    env_dir = os.environ.get("CATEGORY_RULES_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("./rules")


def _tokenize(name_nrm: str):
    # 将字符串按非中文和非字母数字分割，同时保留中文词片段
    parts = re.split(r"[^0-9a-zA-Z\u4e00-\u9fa5]+", name_nrm)
    return [p for p in parts if p]


def map_category_impl(name: str, rules_dir: str = ""):
    """
    返回：(code, matched_token)
    code: 1/2/3/0
    matched_token: 命中的词（便于调试/记录），未命中则空串
    """
    dir_path = _default_rules_dir(rules_dir)
    rules = _load_rules(dir_path)
    key = _normalize(name)

    # 1) 精确匹配（先套装/连衣裙，再下装，再上装，避免歧义）
    for code in (RET_SET, RET_BOTTOM, RET_TOP):
        if key in rules.get(code, set()):
            return code, key

    # 2) 逐 token 匹配（示例： "女装 连衣裙 夏季" -> 命中 "连衣裙"）
    for token in _tokenize(key):
        for code in (RET_SET, RET_BOTTOM, RET_TOP):
            if token in rules.get(code, set()):
                return code, token

    return RET_UNKNOWN, ""


# ----------------------
# ComfyUI Node Definition
# ----------------------
class GarmentCategoryMapper:
    """
    输入：服装第三级分类名称（字符串）、规则目录（可选）、未知时返回值（0/1/2/3）
    输出：整数 code（1/2/3/0）以及命中的 matched_token（便于排查）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "rules_dir": ("STRING", {"multiline": False, "default": ""}),
                "fallback_when_unknown": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("code", "matched_token")
    FUNCTION = "map"
    CATEGORY = "BIMO AI/Clothes"

    def map(self, name, rules_dir="", fallback_when_unknown=0):
        code, matched = map_category_impl(name, rules_dir)
        if code == RET_UNKNOWN and fallback_when_unknown in (0, 1, 2, 3):
            code = fallback_when_unknown
        return (int(code), str(matched))


# 可选：批量处理节点（多行文本 -> 每行一个名称），输出为 JSON 字符串
class GarmentCategoryMapperBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "names_multiline": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "rules_dir": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_result",)
    FUNCTION = "map_batch"
    CATEGORY = "BIMO AI/Clothes"

    def map_batch(self, names_multiline, rules_dir=""):
        lines = [ln.strip() for ln in str(names_multiline).splitlines() if ln.strip()]
        results = []
        for ln in lines:
            code, matched = map_category_impl(ln, rules_dir)
            results.append({"name": ln, "code": int(code), "matched": matched})
        return (json.dumps(results, ensure_ascii=False),)
