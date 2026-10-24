# -*- coding: utf-8 -*-
"""
agent/llm.py —— 本地模型封装 + 可离线 Mock
============================================

Coding Agent 对模型的诉求：能稳定输出「结构化动作 JSON」供主循环解析。
但本地 qwen3:4b 的 JSON 工具调用可能不稳定，所以本模块提供两条路：

    1. OllamaLLM   —— 真调本地 Ollama（qwen3:4b / 更大模型）；
    2. MockLLM     —— 确定性脚本，走完整个 ReAct 循环（离线演示/单测用）。

二者实现同一接口 complete(messages) -> str，主循环无感知切换（依赖倒置）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import re

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    """真·本地模型：通过 Ollama /api/chat 调用，开启 JSON 模式。"""

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL):
        self.model, self.url = model, url

    def complete(self, messages: list[dict], temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",                       # 强制 JSON，降低解析失败率
            "options": {"temperature": temperature},
        }
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise LLMError(f"无法连接 Ollama：{e}。请先 `ollama serve` 并 `ollama pull {self.model}`。") from e
        text = (body.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"模型返回为空：{body}")
        return text.strip()


class MockLLM:
    """确定性 Mock：根据用户任务生成预设动作脚本，用于离线演示/测试。

    它按任务类型模拟「修复/新增/补测试」三类最小改码流程，让主循环的
    解析/分发/回填/终止全链路可被验证，而不依赖真实模型质量。
    """

    def __init__(self):
        self.script: list[dict] | None = None
        self.i = 0

    def _build_script(self, task: str) -> list[dict]:
        """按任务关键词返回不同的工具序列（展示不同工具的使用）。"""
        if re.search(r"修复|fix|add 函数|错误", task, re.I):
            return [
                {"tool": "Read", "args": {"path": "src/math_utils.py"}, "thought": "先读目标文件"},
                {"tool": "Edit", "args": {"path": "src/math_utils.py",
                                          "old_string": "return a - b",
                                          "new_string": "return a + b"}, "thought": "修复 add"},
                {"final": "已修复 add 函数：return a - b → return a + b。"},
            ]
        if re.search(r"新增|加一个|subtract|multiply|divide", task, re.I):
            return [
                {"tool": "Read", "args": {"path": "src/math_utils.py"}, "thought": "读文件末尾"},
                {"tool": "Edit", "args": {"path": "src/math_utils.py",
                                          "old_string": "    return a * b",
                                          "new_string": "    return a * b\n\n\ndef subtract(a, b):\n    \"\"\"两数相减。\"\"\"\n    return a - b"},
                 "thought": "追加 subtract 函数"},
                {"final": "已新增 subtract 函数。"},
            ]
        if re.search(r"测试|test", task, re.I):
            return [
                {"tool": "Write", "args": {"path": "tests/test_math_utils.py",
                                           "content": "from math_utils import add, multiply\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\ndef test_multiply():\n    assert multiply(2, 3) == 6\n"},
                 "thought": "补一个测试文件"},
                {"final": "已补测试 tests/test_math_utils.py。"},
            ]
        return [{"final": "（Mock 无法识别的任务类型）"}]

    def complete(self, messages: list[dict], temperature: float = 0.1) -> str:
        if self.script is None:
            # 从第一条 user 消息里取任务描述
            task = next((m["content"] for m in messages if m.get("role") == "user"), "")
            self.script = self._build_script(task)
            self.i = 0
        if self.i < len(self.script):
            step = self.script[self.i]
            self.i += 1
            return json.dumps(step, ensure_ascii=False)
        return json.dumps({"final": "（脚本已耗尽）"})


def make_llm(use_mock: bool = False) -> object:
    """工厂：按需返回 OllamaLLM 或 MockLLM。"""
    return MockLLM() if use_mock else OllamaLLM()
