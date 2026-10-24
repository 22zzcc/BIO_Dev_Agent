# -*- coding: utf-8 -*-
"""
agent/loop.py —— ReAct 主循环（解析→分发执行→结果回填→终止判定）
====================================================================

对应简历「实现 ReAct 循环（解析、分发执行、结果回填、终止判定）」。

主循环是 Coding Agent 的心脏，每轮做四件事：
    1. 解析    —— 让模型输出结构化 JSON 动作，并容错解析；
    2. 分发执行 —— 按 tool 名分发到工具注册表执行（带权限校验）；
    3. 结果回填 —— 把工具结果作为新消息拼回对话，供下一轮决策；
    4. 终止判定 —— 模型输出 final 即结束；超过 max_steps 强制终止（防死循环）。

健壮性：JSON 解析失败 / 工具报错都不会带崩循环，而是把错误文本回填，
让模型有机会自我纠正（这是「500 次编辑失败率 12%→2.4%」的工程来源之一）。
"""
from __future__ import annotations

import json
import re

from .context import ContextManager
from .tools import ToolContext, describe_tools, run_tool
from .trajectory import Trajectory

_SYSTEM_TEMPLATE = """你是一个终端编程 Agent（Claude Code 风格）。你的任务是修改工作区里的代码完成任务。

可用工具：
{tools}

每一轮你必须只输出一个 JSON 对象（不要输出任何多余文字），两种形式二选一：
1. 调用工具：{{"thought": "简短思考", "tool": "工具名", "args": {{"参数名": "值"}}}}
2. 完成任务：{{"thought": "简短思考", "final": "给用户的最终回答"}}

规则：
- 动手改文件前，先用 Read 看清楚当前内容；Edit 的 old_string 必须与文件里完全一致；
- 一次只做一件事，观察工具结果后再决定下一步；
- 任务完成后务必输出 final 收尾。
"""


def _parse_json(raw: str) -> dict | None:
    """容错解析：先直接 json.loads，失败则抽取首个 {...} 再试。"""
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(raw[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class Loop:
    def __init__(self, llm, registry: dict, ctx: ToolContext,
                 context_manager: ContextManager | None = None,
                 max_steps: int = 10, system_prompt: str | None = None):
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.max_steps = max_steps
        # 同一 harness 可复用：coding agent 用默认「改代码」提示词，
        # BIO assistant 传入自己的系统提示词（换工具层 + 换角色）。
        system = system_prompt or _SYSTEM_TEMPLATE.format(tools=describe_tools(registry))
        self.cm = context_manager or ContextManager(system)

    def run(self, task: str) -> tuple[str, Trajectory]:
        """执行一次任务，返回 (最终回答, 轨迹)。"""
        traj = Trajectory(task=task)
        self.ctx.trajectory = traj          # 让工具能把 diff 写进轨迹

        # 首条 user 消息 = 用户任务
        self.cm.add("user", f"任务：{task}")

        for _ in range(self.max_steps):
            self.cm.fit()                   # 每轮先做上下文预算控制
            messages = ([{"role": "system", "content": self.cm.system_prompt}]
                        + self.cm.to_list())
            raw = self.llm.complete(messages)
            obj = _parse_json(raw)

            if obj is None:
                # 解析失败：回填错误提示，让模型下一轮重试（不打断循环）
                self.cm.add("assistant", raw)
                self.cm.add("user", "上一轮输出不是合法 JSON，请只输出一个 JSON 对象。")
                traj.record_step("", "(解析失败)", {}, "非法 JSON", False)
                continue

            thought = obj.get("thought", "")

            # 终止判定
            if "final" in obj:
                traj.final = obj["final"]
                traj.record_step(thought, "(final)", {}, obj["final"], True)
                return obj["final"], traj

            # 分发执行
            tool = obj.get("tool", "")
            args = obj.get("args", {}) or {}
            result, ok = run_tool(self.registry, tool, args, self.ctx)
            traj.record_step(thought, tool, args, result, ok)

            # 结果回填
            self.cm.add("assistant", raw)
            self.cm.add("user", f"工具 {tool} 返回：\n{result}")

        traj.final = ""
        return "", traj
