# -*- coding: utf-8 -*-
"""
agent/context.py —— token 计数 + 三级上下文压缩
=================================================

对应简历「按 token 计数做三级压缩（结果截断、历史摘要、会话重建），40 个长任务中
因超上下文中断由 11 个降至 2 个」。上下文是 Coding Agent 的「工作记忆」：
改一个真实仓库会累积大量读文件/工具结果，很快撞上下文上限，
所以必须有一个**有预算意识**的上下文管理器。

三级压缩（从轻到重，按需逐级触发）：
    L1 结果截断：把历史里超长的工具结果按上限截断（保留头尾）；
    L2 历史摘要：把最旧的一批消息交给 LLM 压缩成一段摘要，替换原文；
    L3 会话重建：丢弃除 system 与最近 N 轮之外的所有内容，从干净状态继续。
"""
from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """轻量 token 估算：中文按 1 字≈1 token，其余按 4 字符≈1 token。

    不是精确的 tokenizer，但对「预算控制」足够用，且零依赖、可解释。
    """
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + other // 4


class ContextManager:
    """维护对话消息列表，并在超预算时自动做三级压缩。"""

    def __init__(self, system_prompt: str, max_tokens: int = 8000,
                 max_result_chars: int = 2000, keep_recent: int = 6,
                 summarizer=None):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_result_chars = max_result_chars
        self.keep_recent = keep_recent
        self.summarizer = summarizer          # 可选：LLM 摘要函数
        self.messages: list[dict] = []
        self.rebuild_count = 0                # 会话重建次数（评测「上下文丢失」用）

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def total_tokens(self) -> int:
        base = estimate_tokens(self.system_prompt)
        return base + sum(estimate_tokens(m["content"]) for m in self.messages)

    def _truncate(self, text: str) -> str:
        """L1：结果截断——保留头尾，中间标注省略。"""
        if len(text) <= self.max_result_chars:
            return text
        half = self.max_result_chars // 2
        return text[:half] + f"\n...（中间省略 {len(text) - self.max_result_chars} 字符）...\n" + text[-half:]

    def _summarize(self, msgs: list[dict]) -> str:
        """L2：历史摘要——有 LLM 用 LLM，否则退化为「截断拼接」。"""
        blob = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in msgs)
        if self.summarizer is not None:
            try:
                return self.summarizer(blob)
            except Exception:
                pass
        return "（历史摘要）" + self._truncate(blob)[:1000]

    def fit(self) -> None:
        """按预算逐级压缩，直到总 token 回到预算内。"""
        # L1：先截断长工具结果
        for m in self.messages:
            if m["role"] == "tool":
                m["content"] = self._truncate(m["content"])
        if self.total_tokens() <= self.max_tokens:
            return

        # L2：摘要最旧的 60% 消息，压缩为一条 system 附注
        split = int(len(self.messages) * 0.6)
        if split > 0 and len(self.messages) - split >= self.keep_recent:
            summary = self._summarize(self.messages[:split])
            self.messages = ([{"role": "user", "content": summary}]
                             + self.messages[split:])
        if self.total_tokens() <= self.max_tokens:
            return

        # L3：会话重建——只留最近 keep_recent 条
        self.rebuild_count += 1
        if len(self.messages) > self.keep_recent:
            self.messages = self.messages[-self.keep_recent:]

    def to_list(self) -> list[dict]:
        return self.messages
