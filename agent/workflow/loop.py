# -*- coding: utf-8 -*-
"""
workflow/loop.py —— 工作流·编排执行（按计划串行执行组件，记录轨迹）
=====================================================================

工作流的核心：拿到路由计划后，按顺序执行各组件、收集结果、记录轨迹，
最后把多步结果聚合成最终回答。轨迹（prompt/组件/动作/结果）落进记忆供评测与复盘。

对比旧 ReAct-JSON 循环：这里「路由」和「执行」解耦——路由由 router 负责，
执行由 loop 负责；组件异常不中断，逐条记录 ok 标记。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .router import Plan


@dataclass
class Trajectory:
    """一次执行轨迹：request + 每一步的组件/动作/参数/结果/是否成功。"""
    request: str = ""
    steps: list[dict] = field(default_factory=list)
    final: str = ""

    def record(self, step, result: str, ok: bool) -> None:
        self.steps.append({"component": step.component, "action": step.action,
                           "args": step.args, "thought": step.thought,
                           "result": result, "ok": ok})


class Workflow:
    """工作流：把 Plan 执行成结果 + 轨迹。"""

    def __init__(self, registry, router, ctx=None):
        self.registry = registry
        self.router = router
        self.ctx = ctx or {}

    def run(self, request: str) -> tuple[str, Trajectory]:
        plan: Plan = self.router.route(request)
        traj = Trajectory(request=request)

        results: list[str] = []
        for step in plan.steps:
            # action 作为独立字段展示，执行时并回参数里（组件签名统一收 action）
            kwargs = dict(step.args)
            if step.action:
                kwargs.setdefault("action", step.action)
            result, ok = self.registry.run(step.component, self.ctx, **kwargs)
            traj.record(step, result, ok)
            if ok and result:
                results.append(result)

        traj.final = "\n\n".join(results) if results else "（无结果）"
        return traj.final, traj
