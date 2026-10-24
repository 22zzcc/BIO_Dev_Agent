# -*- coding: utf-8 -*-
"""
memory/critic.py —— 记忆·评论家（打分 + badcase 收取 + 反哺）
==============================================================

评论家 = PPO 里的 critic + 多一个「badcase 收取」能力：
    1. judge()   —— 给一轮问答打分（规则 + 可选 LLM 判幻觉）；
    2. collect() —— 把低分样本/用户纠正落成 badcase 存进长期记忆；
    3. feedback()—— 把记忆里的 badcase 路由成偏好对（喂 PPO/DPO）。

这样在线信号变成离线训练资产，Agent 越用越准（自我迭代）。
"""
from __future__ import annotations

import re

from .longterm import LongTermMemory

# 各类 issue 的扣分权重
_PENALTY = {"empty_context": 1.0, "hallucination": 0.6, "recall_miss": 0.5,
            "citation_fail": 0.4, "low_relevance": 0.3, "user_correction": 0.5}


class Critic:
    def __init__(self, memory: LongTermMemory | None = None, llm=None):
        self.memory = memory or LongTermMemory()
        self.llm = llm

    def judge(self, question: str, answer: str, citations: list[str] | None = None,
              hit_count: int = 0) -> dict:
        """评论家打分（1.0 满分，按发现的 issue 扣分）。"""
        issues: list[str] = []
        if hit_count == 0 and answer.strip():
            issues.append("empty_context")
        elif citations is not None:
            n = len(citations)
            for m in re.findall(r"\[(\d+)\]", answer):
                if int(m) > n:
                    issues.append("citation_fail")
                    break
        if self.llm is not None and answer.strip() and citations:
            if self._llm_hallucinate(question, answer):
                issues.append("hallucination")
        score = max(0.0, round(1.0 - sum(_PENALTY.get(i, 0.3) for i in issues), 2))
        return {"score": score, "issues": issues}

    def _llm_hallucinate(self, question: str, answer: str) -> bool:
        try:
            prompt = (f"判断回答是否只能由问题推得、资料无法支持（是=幻觉）：\n"
                      f"问题：{question}\n回答：{answer}\n只回答 是/否：")
            return "是" in self.llm.complete([{"role": "user", "content": prompt}])
        except Exception:
            return False

    def collect(self, question: str, answer: str, citations: list[str] | None = None,
                hit_count: int = 0) -> list[dict]:
        """judge 出的 issue 落库成 badcase。"""
        verdict = self.judge(question, answer, citations, hit_count)
        added = []
        for issue in verdict["issues"]:
            added.append(self.memory.add_badcase(question, answer, issue))
        return added

    def collect_feedback(self, question: str, answer: str, feedback: str) -> dict:
        """用户纠正落库（偏好对来源）。"""
        bc = self.memory.add_badcase(question, answer, "user_correction",
                                     source="feedback", user_feedback=feedback)
        self.memory.add_feedback(question, answer, feedback)
        return bc

    def feedback(self) -> dict:
        """反哺：把长期记忆里的带纠正坏例转成偏好对。"""
        pairs = self.memory.preference_pairs()
        self.memory.save()
        return {"preference_pairs": pairs, "memory": self.memory.stats()}
