# -*- coding: utf-8 -*-
"""
components/rag.py —— RAG 组件（agentic RAG 经接口接入，作为组件不丢）
========================================================================

「agentic RAG 不丢、作为组件」的落点：把 mabwell-agentic-rag 封成统一接口
RAGInterface，再包成 Component，工作流可像调建模组件一样调它。

对外三种能力（action）：
    ask      —— 精确问答（agentic RAG）
    diagnose —— 多因诊断（降精度扩召回，罗列多种可能原因）
    bio      —— 蛋白演化表征 + 药性推导（结合数据库建模）

后端可切换：local（直连本地 mabwell 项目）/ remote（走其 FastAPI）。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from .base import Component

DEFAULT_MABWELL_ROOT = str(Path.home() / "Projects" / "mabwell-agentic-rag")


class RAGInterface:
    """agentic RAG 的可插拔接口（组件与 RAG 实现解耦）。"""

    def __init__(self, backend: str = "local", mabwell_root: str | None = None,
                 base_url: str = "http://127.0.0.1:8000", use_llm: bool = False):
        self.backend = backend
        self.mabwell_root = mabwell_root or os.environ.get("MABWELL_ROOT", DEFAULT_MABWELL_ROOT)
        self.base_url = base_url
        self.use_llm = use_llm
        self._cache: dict = {}

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

    def ask(self, question: str) -> dict:
        if self.backend == "remote":
            return self._remote("/ask", {"question": question})
        r = self._local("agent").run(question)
        return {"answer": r.answer, "citations": r.citations, "sub_queries": r.sub_queries}

    def diagnose(self, question: str) -> dict:
        if self.backend == "remote":
            return self._remote("/diagnose", {"question": question})
        r = self._local("diagnoser").diagnose(question)
        return {"answer": r.answer, "recall_count": r.recall_count, "citations": r.citations}

    def bio(self, request: str) -> dict:
        if self.backend == "remote":
            return self._remote("/bio", {"request": request})
        r = self._local("bio").run(request)
        return {"answer": r.answer, "properties": r.properties,
                "conservation": {k: v for k, v in r.conservation.items() if k != "profile"},
                "citations": r.citations}

    def feedback(self, question: str, answer: str, user_feedback: str) -> dict:
        critic = self._local("critic")
        critic.collect_user_feedback(question, answer, user_feedback)
        fb = critic.feedback_to_datasets()
        critic.memory.save()
        return {"preference_pairs": len(fb["preference_pairs"]), "memory": critic.memory.stats()}

    def _remote(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _format_citations(cites: list[str] | None, limit: int = 8) -> str:
    if not cites:
        return ""
    return "\n\n参考来源：\n" + "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(cites[:limit]))


def build_rag_component(rag: RAGInterface) -> Component:
    """构建「RAG」组件。"""
    def _run(ctx, action: str, query: str):
        try:
            if action == "ask":
                r = rag.ask(query)
                body = r.get("answer", "")
            elif action == "diagnose":
                r = rag.diagnose(query)
                body = f"（已扩召回 {r.get('recall_count', 0)} 条候选）\n" + r.get("answer", "")
            elif action == "bio":
                r = rag.bio(query)
                body = r.get("answer", "")
            else:
                return f"错误：未知 action「{action}」，可选 ask / diagnose / bio", False
            body += _format_citations(r.get("citations"))
            return body, True
        except Exception as e:
            return f"RAG 组件调用失败：{type(e).__name__}: {e}", False

    return Component(
        name="rag",
        description="agentic RAG：ask=精确问答 / diagnose=多因诊断 / bio=蛋白药性推导",
        params={"action": "ask|diagnose|bio", "query": "问题或请求"},
        func=_run,
    )


def build_feedback_component(rag: RAGInterface) -> Component:
    """构建「反馈」组件（写长期记忆）。"""
    def _run(ctx, question: str, answer: str, feedback: str):
        try:
            r = rag.feedback(question, answer, feedback)
            return (f"已记录反馈：新偏好对 {r.get('preference_pairs', 0)} 条；"
                    f"记忆累计 {r.get('memory', {})}"), True
        except Exception as e:
            return f"反馈记录失败：{type(e).__name__}: {e}", False

    return Component(
        name="feedback",
        description="把用户对某轮回答的纠正写入长期记忆",
        params={"question": "原问题", "answer": "当时的回答", "feedback": "用户纠正"},
        func=_run,
    )
