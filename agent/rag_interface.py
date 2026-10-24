# -*- coding: utf-8 -*-
"""
agent/rag_interface.py —— 把 agentic RAG 封装成 BIO assistant 的一个组件（接口）
================================================================================

这是「agentic RAG 不要丢、作为 BIO assistant 的一个组件」的落点：

    一个统一接口 RAGInterface，对外暴露 4 个能力：
      ask()      —— 精确问答（agentic RAG）
      diagnose() —— 多因诊断（降精度扩召回，罗列多种可能原因）
      bio()      —— 蛋白演化表征 + 药性推导（BioAgent）
      feedback() —— 用户反馈落库（长期记忆 feedback 闭环）

BIO assistant 的工具层只认这个接口，与 RAG 的实现解耦——这就是「设定一个接口」的意义：
换 RAG 实现（本地直连 / FastAPI 远程 / 换成别的检索后端）时，BIO assistant 一行都不用改。

后端可切换：
    - local  ：直接 import 本地 mabwell-agentic-rag 的 core（同机、零网络、秒级）；
    - remote ：走它的 FastAPI（http://127.0.0.1:8000，/ask /diagnose /bio）。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

# 默认 mabwell-agentic-rag 的位置（可用环境变量 MABWELL_ROOT 覆盖）
DEFAULT_MABWELL_ROOT = str(Path.home() / "Projects" / "mabwell-agentic-rag")


class RAGInterface:
    """agentic RAG 的可插拔接口。"""

    def __init__(self, backend: str = "local",
                 mabwell_root: str | None = None,
                 base_url: str = "http://127.0.0.1:8000",
                 use_llm: bool = False):
        self.backend = backend
        self.mabwell_root = mabwell_root or os.environ.get("MABWELL_ROOT", DEFAULT_MABWELL_ROOT)
        self.base_url = base_url
        self.use_llm = use_llm
        self._cache: dict = {}

    # ---- 本地后端：惰性 import mabwell-agentic-rag 的 core ----
    def _local(self, name: str):
        if name not in self._cache:
            sys.path.insert(0, self.mabwell_root)
            from core.ingest import load_index
            index = load_index()
            if name == "agent":
                from core.agent import AgenticRAG
                self._cache[name] = AgenticRAG(index, use_llm=self.use_llm)
            elif name == "diagnoser":
                from core.diagnoser import Diagnoser
                self._cache[name] = Diagnoser(index, use_llm=self.use_llm)
            elif name == "bio":
                from core.bio_agent import BioAgent
                self._cache[name] = BioAgent(index, use_llm=self.use_llm)
            elif name == "critic":
                from core.memory import Critic
                self._cache[name] = Critic(index)
            else:
                raise ValueError(f"未知组件：{name}")
        return self._cache[name]

    # ---- 四个对外能力 ----
    def ask(self, question: str) -> dict:
        """精确问答（agentic RAG）。"""
        if self.backend == "remote":
            return self._remote("/ask", {"question": question})
        r = self._local("agent").run(question)
        return {"answer": r.answer, "citations": r.citations,
                "sub_queries": r.sub_queries}

    def diagnose(self, question: str) -> dict:
        """多因诊断（罗列多种可能原因）。"""
        if self.backend == "remote":
            return self._remote("/diagnose", {"question": question})
        r = self._local("diagnoser").diagnose(question)
        return {"answer": r.answer, "recall_count": r.recall_count,
                "citations": r.citations}

    def bio(self, request: str) -> dict:
        """蛋白演化表征 + 药性推导。"""
        if self.backend == "remote":
            return self._remote("/bio", {"request": request})
        r = self._local("bio").run(request)
        return {"answer": r.answer, "properties": r.properties,
                "conservation": {k: v for k, v in r.conservation.items() if k != "profile"},
                "citations": r.citations}

    def feedback(self, question: str, answer: str, user_feedback: str) -> dict:
        """用户反馈落库（长期记忆闭环）。"""
        critic = self._local("critic")
        critic.collect_user_feedback(question, answer, user_feedback)
        fb = critic.feedback_to_datasets()
        critic.memory.save()
        return {"preference_pairs": len(fb["preference_pairs"]),
                "memory": critic.memory.stats()}

    # ---- 远程后端：走 FastAPI ----
    def _remote(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
