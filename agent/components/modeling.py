# -*- coding: utf-8 -*-
"""
components/modeling.py —— 建模组件（蛋白分子建模，纯计算、零 RAG 依赖）
========================================================================

「建模模式」的本体：给一条/一组氨基酸序列，算出「基本蛋白分子模型」——
    1. protein_properties() —— 分子量 MW / 等电点 pI / 疏水性 GRAVY / 氨基酸组成；
    2. conservation_profile() —— 多序列位点保守性（Shannon 熵）→ 演化表征与趋向。

纯 Python 实现，不依赖 Biopython，离线秒级。这是「独立于 RAG 的建模能力」：
即便没有知识库，也能对序列做建模；结合 rag 组件（药性推导）才是完整闭环。
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .base import Component

AA = "ACDEFGHIKLMNPQRSTVWY"
_SEQ_RE = re.compile(rf"[{AA}]{{{8},}}")   # 提取序列：>=8 个连续标准氨基酸

# 残基平均分子量（Da）
_AA_MW = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}
# Kyte-Doolittle 疏水性标度
_AA_HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}
# 可解离基团 pKa（用于等电点）
_PKA_NTERM, _PKA_CTERM = 9.69, 2.34
_PKA_POS = {"K": 10.53, "R": 12.48, "H": 6.04}
_PKA_NEG = {"D": 3.86, "E": 4.25, "C": 8.33, "Y": 10.07}


# ---- 纯计算函数 ----
def extract_sequences(text: str) -> list[str]:
    """从文本里提取氨基酸序列（>=8 个连续标准氨基酸），去重保序。"""
    return list(dict.fromkeys(_SEQ_RE.findall(text)))


def protein_properties(seq: str) -> dict:
    """单条序列的基本模型：分子量 / 等电点 / 疏水性 / 氨基酸组成。"""
    seq = "".join(c for c in seq.upper() if c in AA)
    n = len(seq)
    if n == 0:
        return {}
    comp = {aa: seq.count(aa) for aa in AA if seq.count(aa)}
    mw = sum(_AA_MW[aa] for aa in seq) - 18.0153 * (n - 1)
    gravy = sum(_AA_HYDRO[aa] for aa in seq) / n
    pi = _compute_pi(seq)
    return {"length": n, "mw_da": round(mw, 1), "pi": round(pi, 2),
            "gravy": round(gravy, 3), "composition": comp}


def _net_charge(seq: str, ph: float) -> float:
    charge = 1.0 / (1.0 + 10 ** (ph - _PKA_NTERM))
    charge -= 1.0 / (1.0 + 10 ** (_PKA_CTERM - ph))
    for aa in seq:
        if aa in _PKA_POS:
            charge += 1.0 / (1.0 + 10 ** (ph - _PKA_POS[aa]))
        elif aa in _PKA_NEG:
            charge -= 1.0 / (1.0 + 10 ** (_PKA_NEG[aa] - ph))
    return charge


def _compute_pi(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def conservation_profile(sequences: list[str]) -> dict:
    """等长对齐序列的位点保守性（Shannon 熵 → 保守分数）。

    演化表征 = 每个位点的保守分数序列；
    演化趋向 = 保守位点（功能约束）vs 可变位点（演化适应）。
    """
    if not sequences:
        return {}
    L = min(len(s) for s in sequences)
    seqs = [s[:L] for s in sequences]
    n = len(seqs)
    hmax = math.log2(min(20, n)) if n > 1 else 1.0
    profile = []
    for i in range(L):
        col = [s[i] for s in seqs]
        counts = Counter(col)
        h = -sum((c / n) * math.log2(c / n) for c in counts.values())
        cons = 1 - h / hmax
        profile.append({"pos": i + 1, "consensus": max(counts, key=counts.get),
                        "entropy": round(h, 3), "conservation": round(cons, 3)})
    conserved = [p["pos"] for p in profile if p["conservation"] >= 0.8]
    variable = [p["pos"] for p in profile if p["conservation"] <= 0.5]
    return {"length": L, "n_seqs": n,
            "conserved_positions": conserved, "variable_positions": variable,
            "profile": profile}


# ---- 组件化封装 ----
def _action_properties(ctx, sequence: str):
    """建模·基本模型：给定序列算理化性质。"""
    props = protein_properties(sequence)
    if not props:
        return "错误：未能从输入中解析到有效序列", False
    comp = ", ".join(f"{k}={v}" for k, v in props["composition"].items())
    return (f"长度 {props['length']}；分子量 {props['mw_da']} Da；pI {props['pi']}；"
            f"GRAVY {props['gravy']}\n氨基酸组成：{comp}", True)


def _action_conservation(ctx, sequences):
    """建模·演化表征：多序列保守性分析。"""
    seqs = list(sequences) if isinstance(sequences, list) else extract_sequences(str(sequences))
    if len(seqs) < 2:
        return "错误：保守性分析至少需要 2 条等长序列", False
    r = conservation_profile(seqs)
    return (f"对齐长度 {r['length']} / {r['n_seqs']} 条序列；"
            f"保守位点 {len(r['conserved_positions'])} 个 / 可变位点 {len(r['variable_positions'])} 个",
            True)


def build_modeling_component() -> Component:
    """构建「建模」组件。"""
    return Component(
        name="modeling",
        description="蛋白分子建模：理化性质(properties) / 演化保守性(conservation)，纯计算",
        params={"action": "properties|conservation", "sequence": "序列或序列列表"},
        func=lambda ctx, action, sequence:
            _action_properties(ctx, sequence) if action == "properties"
            else _action_conservation(ctx, sequence),
    )
