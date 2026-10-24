# -*- coding: utf-8 -*-
"""
agent/llm.py —— 模型接口（三要素共用的底座）
=============================================

工作流（路由/推导）、记忆（评论家判幻觉）都可能用到本地模型，统一走这里。
默认 Ollama；不可用时给出「ollama pull」指引，上层各自降级（规则路由/模板作答）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    """本地 Ollama 模型（JSON 模式）。"""

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL):
        self.model, self.url = model, url

    def complete(self, messages: list[dict], temperature: float = 0.1) -> str:
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "format": "json", "options": {"temperature": temperature},
        }
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise LLMError(f"无法连接 Ollama：{e}。请先 `ollama serve` 并 `ollama pull {self.model}`。") from e
        text = (body.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"模型返回为空：{body}")
        return text.strip()
