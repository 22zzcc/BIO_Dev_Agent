# -*- coding: utf-8 -*-
"""
memory/longterm.py —— 记忆·长期（badcase / 反馈落库，跨会话沉淀）
=================================================================

长期记忆 = 把「坏例 + 用户反馈」持久化，让在线信号变成离线资产：
    - 上下文自动坏例（引用越界/漏召回/低相关/零召回）
    - 对话反馈（用户 👎 / 纠正）

这些数据三处复用：回归集（eval）、LoRA 三元组、PPO 偏好对（见 critic.py 反哺）。
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

_ISSUE_WHY = {
    "citation_fail": "生成：引用越界/错引",
    "recall_miss": "检索：漏召回",
    "low_relevance": "检索：召回噪声",
    "empty_context": "检索：零召回",
    "hallucination": "生成：幻觉",
    "user_correction": "生成：用户纠正",
}


class LongTermMemory:
    """长期记忆：badcase + 反馈的 JSON 持久化。"""

    def __init__(self, path: str | Path = "data/memory.json"):
        self.path = Path(path)
        self.badcases: list[dict] = []
        self.feedback: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.badcases = d.get("badcases", [])
            self.feedback = d.get("feedback", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"badcases": self.badcases, "feedback": self.feedback},
            ensure_ascii=False, indent=2), encoding="utf-8")

    def add_badcase(self, question: str, answer: str, issue: str,
                    source: str = "context", user_feedback: str = "") -> dict:
        bc = {"question": question, "answer": answer, "issue": issue,
              "source": source, "why": _ISSUE_WHY.get(issue, "其他"),
              "user_feedback": user_feedback, "ts": time.time()}
        self.badcases.append(bc)
        return bc

    def add_feedback(self, question: str, answer: str, feedback: str) -> None:
        self.feedback.append({"question": question, "answer": answer,
                              "feedback": feedback, "ts": time.time()})

    def preference_pairs(self) -> list[dict]:
        """从带纠正的 badcase 提取 (prompt, chosen, rejected) 偏好对（喂 PPO/DPO）。"""
        pairs = []
        for b in self.badcases:
            if b.get("user_feedback"):
                pairs.append({"prompt": b["question"], "rejected": b["answer"],
                              "chosen": b["user_feedback"]})
        return pairs

    def stats(self) -> dict:
        return {"n_badcases": len(self.badcases), "n_feedback": len(self.feedback),
                "by_issue": dict(Counter(b["issue"] for b in self.badcases))}
