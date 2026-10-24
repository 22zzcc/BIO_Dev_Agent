#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py —— Coding Agent CLI 入口
=================================

用法（在项目根目录下运行）：

    # 离线演示（确定性 Mock 模型，走完整 ReAct 循环）
    python main.py run "把 math_utils.py 里的 add 函数修好" --mock

    # 真调本地 Ollama（qwen3:4b）
    python main.py run "给项目加一个 subtract 函数" --level write

    # 一键评测（Mock 跑一批任务，输出成功率与失败分类）
    python main.py eval --mock

参数：
    --root   工作区根目录（默认 demo_repo）
    --level  权限档 read/write/execute（默认 write）
    --mock   使用 MockLLM（离线演示）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_agent(args):
    from agent.llm import make_llm
    from agent.permissions import Permissions
    from agent.tools import ToolContext, build_registry
    from agent.loop import Loop
    from agent.router import Router

    root = Path(args.root).resolve()
    perms = Permissions(level=args.level, workspace_root=root)
    registry = build_registry()
    llm = make_llm(args.mock)

    ctx = ToolContext(root=root, permissions=perms)
    router = Router(llm, registry, ctx)
    ctx.router = router                       # 让 Task 工具能派生子 Agent

    loop = Loop(llm, registry, ctx, max_steps=args.max_steps)
    return loop, perms, router


def cmd_run(args) -> None:
    loop, perms, router = build_agent(args)
    print(f"[工作区] {Path(args.root).resolve()}    [权限档] {perms.level}")
    print("=" * 70)
    final, traj = loop.run(args.task)

    print(f"最终回答：\n{final or '(未在步数内完成)'}\n")
    print("轨迹：")
    for i, s in enumerate(traj.steps, start=1):
        print(f"  {i}. [{s['tool']}] {'✓' if s['ok'] else '✗'} {s['thought'] or s['result'][:60]}")
    if traj.diffs:
        print("变更(diff)：")
        for d in traj.diffs:
            print(f"  {d['path']}: {d['old'][:40]!r} -> {d['new'][:40]!r}")
    print(f"[路由统计] 规则 {router.rule_hits} / 向量 {router.vector_hits} / LLM {router.llm_hits}（LLM 占比 {router.llm_ratio:.0%}）")


def cmd_eval(args) -> None:
    from agent.trajectory import evaluate

    tasks = [
        "修复 add 函数的错误",
        "新增一个 subtract 函数",
        "给现有代码补一个测试",
    ]

    def reset_workspace():
        """每个任务前把工作区恢复到「带 bug」的初始态，保证任务互不影响。"""
        src = Path(args.root) / "src" / "math_utils.py"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(
            "# 数学工具函数\n\n\n"
            "def add(a, b):\n"
            "    \"\"\"两数相加（注意：这里故意埋了一个 bug）。\"\"\"\n"
            "    return a - b\n\n\n"
            "def multiply(a, b):\n"
            "    \"\"\"两数相乘。\"\"\"\n"
            "    return a * b\n",
            encoding="utf-8")
        # 清理可能残留的测试文件
        t = Path(args.root) / "tests" / "test_math_utils.py"
        if t.exists():
            t.unlink()

    def run_one(task):
        reset_workspace()
        loop, perms, router = build_agent(args)
        final, traj = loop.run(task)
        return traj, (loop.cm.rebuild_count > 0)

    result = evaluate(tasks, run_one)
    print(f"成功率：{result.success}/{result.total} = {result.success_rate:.1%}")
    for f in result.failures:
        print(f"  失败：{f['task']}  → 归类「{f['type']}」")


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code 式 Coding Agent MVP")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="执行一个改码任务")
    p_run.add_argument("task", help="任务描述")
    p_run.add_argument("--root", default="demo_repo", help="工作区根目录")
    p_run.add_argument("--level", default="write", choices=["read", "write", "execute"])
    p_run.add_argument("--mock", action="store_true", help="使用 MockLLM 离线演示")
    p_run.add_argument("--max-steps", type=int, default=10)
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="跑一批任务评测成功率")
    p_eval.add_argument("--root", default="demo_repo")
    p_eval.add_argument("--level", default="write", choices=["read", "write", "execute"])
    p_eval.add_argument("--mock", action="store_true")
    p_eval.add_argument("--max-steps", type=int, default=10)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
