# -*- coding: utf-8 -*-
"""
agent/trajectory.py —— 轨迹记录与评测
========================================

对应简历「记录 prompt、工具调用与 diff，120 个真实任务成功率 76.7%，
较一次性生成 patch 基线 +24pp，定位工具误选、编辑失效、上下文丢失三类失败」。

轨迹（trajectory）是 Agent 的「黑匣子」：出了问题能复盘「它当时想了什么、
调了什么工具、改了哪个文件」，从而把失败归类、反哺改进。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """一次任务的完整轨迹。"""
    task: str = ""
    steps: list[dict] = field(default_factory=list)   # {thought, tool, args, result, ok}
    diffs: list[dict] = field(default_factory=list)   # {path, old, new}
    final: str = ""

    def record_step(self, thought: str, tool: str, args: dict,
                    result: str, ok: bool) -> None:
        self.steps.append({"thought": thought, "tool": tool, "args": args,
                           "result": result[:500], "ok": ok})

    def record_diff(self, path: str, old: str, new: str) -> None:
        self.diffs.append({"path": path, "old": old, "new": new})

    def to_dict(self) -> dict:
        return {"task": self.task, "steps": self.steps, "diffs": self.diffs,
                "final": self.final, "n_steps": len(self.steps),
                "n_diffs": len(self.diffs)}


# ---------------------------------------------------------------------------
# 失败分类：把「没成功」归到三个根因之一
# ---------------------------------------------------------------------------
def classify_failure(traj: Trajectory, hit_rebuild: bool = False) -> str:
    """按简历三类失败对一次任务定位根因。

    - 工具误选：某步调用后 ok=False 且错误里含「参数/未知工具/权限档拒绝」；
    - 编辑失效：Edit 步 ok=False（old_string 未命中或多处命中）；
    - 上下文丢失：发生会话重建 / 步数用尽仍无 final。
    """
    for s in traj.steps:
        if not s["ok"]:
            if s["tool"] == "Edit" and ("匹配" in s["result"] or "命中" in s["result"]):
                return "编辑失效"
            if s["tool"] == "Bash" and "权限" in s["result"]:
                return "权限拦截"
            return "工具误选"
    if hit_rebuild or not traj.final:
        return "上下文丢失"
    return "其他"


@dataclass
class EvalResult:
    success: int = 0
    total: int = 0
    failures: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0


def evaluate(tasks: list[str], run_one) -> EvalResult:
    """对一批任务跑 Agent，统计成功率并按三类失败归因。

    :param tasks:   任务描述列表
    :param run_one: callable(task) -> (Trajectory, hit_rebuild)
    """
    res = EvalResult(total=len(tasks))
    for task in tasks:
        traj, hit_rebuild = run_one(task)
        ok = bool(traj.final) and all(s["ok"] for s in traj.steps)
        if ok:
            res.success += 1
        else:
            res.failures.append({"task": task, "type": classify_failure(traj, hit_rebuild)})
    return res
