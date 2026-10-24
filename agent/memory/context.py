# -*- coding: utf-8 -*-
"""
memory/context.py —— 记忆·短期（token 计数 + 三级上下文压缩）
=============================================================

短期记忆 = Agent 的「工作记忆」：多步执行会累积组件结果，很快撞上下文上限，
所以需要预算控制。三级压缩（从轻到重，逐级触发）：
    L1 结果截断 —— 超长的组件结果按上限截断（保留头尾）；
    L2 历史摘要 —— 把最旧一批消息压缩成摘要；
    L3 会话重建 —— 丢弃除最近 N 轮外的内容。
"""
from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """轻量 token 估算：中文 1 字≈1 token，其余 4 字符≈1 token。"""
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + other // 4


class ShortTermMemory:
    """短期记忆：维护消息列表，超预算自动三级压缩。"""

    def __init__(self, max_tokens: int = 8000, max_result_chars: int = 2000,
                 keep_recent: int = 6):
        self.max_tokens = max_tokens
        self.max_result_chars = max_result_chars
        self.keep_recent = keep_recent
        self.messages: list[dict] = []
        self.rebuild_count = 0

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def total_tokens(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_result_chars:
            return text
        half = self.max_result_chars // 2
        return (text[:half] + f"\n…（省略 {len(text) - self.max_result_chars} 字符）…\n"
                + text[-half:])

    def fit(self) -> None:
        """按预算逐级压缩。"""
        for m in self.messages:
            if m["role"] == "tool":
                m["content"] = self._truncate(m["content"])
        if self.total_tokens() <= self.max_tokens:
            return
        split = int(len(self.messages) * 0.6)
        if split > 0 and len(self.messages) - split >= self.keep_recent:
            blob = "\n".join(m["content"][:200] for m in self.messages[:split])
            self.messages = [{"role": "user", "content": "（历史摘要）" + self._truncate(blob)[:1000]}
                             ] + self.messages[split:]
        if self.total_tokens() <= self.max_tokens:
            return
        self.rebuild_count += 1
        if len(self.messages) > self.keep_recent:
            self.messages = self.messages[-self.keep_recent:]
