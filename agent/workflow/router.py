# -*- coding: utf-8 -*-
"""
workflow/router.py —— 工作流·意图路由（建模/诊断/问答 → 组件+动作）
====================================================================

把自然语言请求路由成「组件 + 动作 + 参数」的执行计划。级联策略（面试可讲）：
    1. 规则（快、离线）：关键词 + 显式序列检测；
    2. LLM（可选兜底）：规则不自信时让模型决定。
默认规则层足够覆盖三大意图，零 LLM 也能离线跑通。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PlanStep:
    component: str
    action: str
    args: dict
    thought: str


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)

    def add(self, component, action, args, thought) -> "Plan":
        self.steps.append(PlanStep(component, action, args, thought))
        return self


class Router:
    """意图路由：规则优先，LLM 兜底。"""

    def __init__(self, llm=None):
        self.llm = llm

    def route(self, request: str) -> Plan:
        plan = self._rule_route(request)
        if plan is not None:
            return plan
        if self.llm is not None:
            return self._llm_route(request)
        # 默认：精确问答
        return Plan().add("rag", "ask", {"query": request}, "事实类问题，精确问答")

    # ---- 规则层 ----
    def _rule_route(self, request: str) -> Plan | None:
        # ① 显式序列 → 建模组件（纯计算，无需 RAG）
        from ..components.modeling import extract_sequences
        seqs = extract_sequences(request)
        if seqs:
            lengths = {len(s) for s in seqs}
            if len(seqs) >= 2 and len(lengths) == 1:
                return Plan().add("modeling", "conservation", {"sequence": seqs},
                                  "检测到多条等长序列，做演化保守性分析")
            return Plan().add("modeling", "properties", {"sequence": seqs[0]},
                              "检测到序列，做基本分子建模")

        # ② 诊断类 → RAG·多因诊断
        if re.search(r"诊断|原因|为什么|哪些|偏低|风险|问题", request):
            return Plan().add("rag", "diagnose", {"query": request},
                              "诊断类问题，用多因诊断")

        # ③ 建模/蛋白分析 → RAG·药性推导（建模模式）
        if re.search(r"建模|模型|蛋白|序列|演化|药性|抗体|亲和力|稳定性|分子|靶点", request):
            return Plan().add("rag", "bio", {"query": request},
                              "涉及蛋白建模/药性，走结合数据库的推导")

        return None  # 规则不自信，交给 LLM 或默认

    # ---- LLM 兜底 ----
    def _llm_route(self, request: str) -> Plan:
        try:
            import json
            prompt = ("把请求归类：建模/诊断/问答，只输出 JSON "
                      '{"action":"bio|diagnose|ask","query":"改写后的查询"}。\n'
                      f"请求：{request}\nJSON：")
            raw = self.llm.complete([{"role": "user", "content": prompt}])
            obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            action = obj.get("action", "ask")
            return Plan().add("rag", action, {"query": obj.get("query", request)},
                              f"LLM 路由到 {action}")
        except Exception:
            return Plan().add("rag", "ask", {"query": request}, "LLM 路由失败，退回问答")
