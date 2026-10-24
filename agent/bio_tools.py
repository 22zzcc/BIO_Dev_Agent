# -*- coding: utf-8 -*-
"""
agent/bio_tools.py —— BIO assistant 的工具层（复用 coding agent 的 harness）
============================================================================

「将 coding agent 改设为 BIO assistant」的关键一步：**同一个 harness，换工具层**。
coding agent 的工具是 Read/Write/Edit/Glob/Grep/Bash/Task（操作文件）；
BIO assistant 的工具是：

    rag       —— 调用 agentic RAG 组件（ask=精确问答 / diagnose=多因诊断 / bio=蛋白药性推导）
    feedback  —— 把用户对某轮回答的纠正写入长期记忆（数据闭环）

工具实现走 RAGInterface（见 rag_interface.py），BIO assistant 与 RAG 实现解耦。
主循环（ReAct）、权限、上下文、路由全部复用 coding agent 的 agent/loop.py 等。
"""
from __future__ import annotations

import json
import re

from .rag_interface import RAGInterface
from .tools import Tool

# 工具 → 所需权限档（rag 只读、feedback 写记忆）
BIO_TOOL_LEVEL = {"rag": "read", "feedback": "write"}


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
def _tool_rag(_ctx, rag: RAGInterface, action: str, query: str) -> tuple[str, bool]:
    """调用 agentic RAG 组件，按 action 分发三种能力。"""
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

        cites = r.get("citations") or []
        if cites:
            body += "\n\n参考来源：\n" + "\n".join(
                f"[{i + 1}] {c}" for i, c in enumerate(cites[:8]))
        return body, True
    except Exception as e:  # RAG 组件异常不能带崩主循环
        return f"RAG 组件调用失败：{type(e).__name__}: {e}", False


def _tool_feedback(_ctx, rag: RAGInterface, question: str, answer: str,
                   feedback: str) -> tuple[str, bool]:
    """用户纠正落库：沉淀成偏好对，喂 PPO/DPO（长期记忆闭环）。"""
    try:
        r = rag.feedback(question, answer, feedback)
        return (f"已记录反馈：新偏好对 {r.get('preference_pairs', 0)} 条；"
                f"记忆累计 {r.get('memory', {})}"), True
    except Exception as e:
        return f"反馈记录失败：{type(e).__name__}: {e}", False


def build_bio_registry(rag: RAGInterface) -> dict[str, Tool]:
    """构建 BIO assistant 的工具注册表（与 coding agent 的 build_registry 同构）。"""
    return {
        "rag": Tool(
            "rag", "调用 agentic RAG 组件",
            {"action": "ask|diagnose|bio（精确问答/多因诊断/蛋白药性推导）", "query": "问题或请求"},
            lambda ctx, action, query: _tool_rag(ctx, rag, action, query)),
        "feedback": Tool(
            "feedback", "把用户对某轮回答的纠正写入长期记忆",
            {"question": "原问题", "answer": "当时的回答", "feedback": "用户纠正/更好答案"},
            lambda ctx, question, answer, feedback:
                _tool_feedback(ctx, rag, question, answer, feedback)),
    }


# ---------------------------------------------------------------------------
# 离线演示：确定性路由 Mock（无需真实 LLM 也能走完 ReAct 循环）
# ---------------------------------------------------------------------------
class BioMockLLM:
    """按关键词路由到 rag 的不同动作，走完 ReAct 循环（离线演示/测试用）。"""

    def __init__(self):
        self.script: list[dict] | None = None
        self.i = 0

    def _route(self, task: str) -> list[dict]:
        t = task.replace("任务：", "")
        if re.search(r"诊断|原因|为什么|哪些|偏低", t):
            return [{"thought": "这是诊断类问题，用多因诊断", "tool": "rag",
                     "args": {"action": "diagnose", "query": t}},
                    {"final": "（已完成多因诊断，结果见上）"}]
        if re.search(r"蛋白|序列|演化|药性|分子|抗体|亲和力|稳定性", t):
            return [{"thought": "涉及蛋白分析，用 bio 推导", "tool": "rag",
                     "args": {"action": "bio", "query": t}},
                    {"final": "（已完成蛋白药性推导，结果见上）"}]
        return [{"thought": "事实类问题，用精确问答", "tool": "rag",
                 "args": {"action": "ask", "query": t}},
                {"final": "（已完成问答，结果见上）"}]

    def complete(self, messages: list[dict], temperature: float = 0.1) -> str:
        if self.script is None:
            task = next((m["content"] for m in messages if m.get("role") == "user"), "")
            self.script = self._route(task)
            self.i = 0
        if self.i < len(self.script):
            step = self.script[self.i]
            self.i += 1
            return json.dumps(step, ensure_ascii=False)
        return json.dumps({"final": "（结束）"})
