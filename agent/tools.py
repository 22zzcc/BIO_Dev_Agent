# -*- coding: utf-8 -*-
"""
agent/tools.py —— 七个核心工具（Read/Write/Edit/Glob/Grep/Bash/Task）
=====================================================================

对应简历「封装 Read/Write/Edit/Glob/Grep/Bash/Task 等 7 个核心工具，
500 次编辑操作失败率由 12% 降至 2.4%」。

工具是 Agent 与真实文件系统之间的**唯一通道**：Agent 不能直接碰文件，
必须通过工具 + 权限校验。这样做的价值：
    - 每次读写都落在权限与路径约束内（见 permissions.py）；
    - 每个动作都被轨迹记录（见 trajectory.py），可审计、可回放、可评测。

Edit 的「失败率」来自三类错误，本实现逐一处理并给出明确报错：
    1. old_string 找不到（编辑失效）—— 报「未匹配到目标文本」；
    2. old_string 命中多处（歧义）—— 报「命中 N 处，请提供更长上下文」；
    3. 目标路径越界/无写权限 —— 被 permissions 拦截。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .permissions import Permissions, PermissionError_


# ---------------------------------------------------------------------------
# 工具上下文：把「工作区根 / 权限 / 子Agent路由」传给每个工具
# ---------------------------------------------------------------------------
@dataclass
class ToolContext:
    root: Path
    permissions: Permissions
    router: object | None = None       # 供 Task 工具派生子 Agent
    trajectory: object | None = None   # 供记录 diff / 调用


# ---------------------------------------------------------------------------
# 工具实现（每个函数返回统一结构 (文本结果, 是否成功)）
# ---------------------------------------------------------------------------
def _read(ctx: ToolContext, path: str, start: int | None = None, end: int | None = None) -> tuple[str, bool]:
    """读取文件，可选行区间；带行号输出，方便后续 Edit 定位。"""
    p = ctx.permissions.resolve_path(path)
    if not p.exists():
        return f"错误：文件不存在 {path}", False
    lines = p.read_text(encoding="utf-8").splitlines()
    start = max(1, start or 1)
    end = min(len(lines), end or len(lines))
    out = [f"{i:>4} | {lines[i-1]}" for i in range(start, end + 1)]
    return "\n".join(out) or "(空文件)", True


def _write(ctx: ToolContext, path: str, content: str) -> tuple[str, bool]:
    """整文件写入（创建或覆盖）。"""
    if not ctx.permissions.tool_allowed("Write"):
        return "错误：当前权限档不允许写入（需 write 档）", False
    p = ctx.permissions.resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字符）", True


def _edit(ctx: ToolContext, path: str, old_string: str, new_string: str) -> tuple[str, bool]:
    """精确字符串替换（aider 风格）：old_string 必须且只能命中一次。"""
    if not ctx.permissions.tool_allowed("Edit"):
        return "错误：当前权限档不允许编辑（需 write 档）", False
    p = ctx.permissions.resolve_path(path)
    if not p.exists():
        return f"错误：文件不存在 {path}", False
    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        # 「编辑失效」主因之一：模型给的上下文与文件实际内容不一致
        return f"错误：未在 {path} 中匹配到目标文本（请用 Read 查看当前内容后再改）", False
    if count > 1:
        return f"错误：目标文本命中 {count} 处，请提供更长、更唯一的上下文", False
    new_text = text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")
    # 记录 diff（轨迹评测用）
    if ctx.trajectory is not None:
        ctx.trajectory.record_diff(path, old_string, new_string)
    return f"已修改 {path}（1 处替换）", True


def _glob(ctx: ToolContext, pattern: str) -> tuple[str, bool]:
    """按 glob 模式列文件（相对 workspace 根）。"""
    matches = sorted(str(p.relative_to(ctx.root)) for p in ctx.root.glob(pattern)
                     if p.is_file())
    return "\n".join(matches) if matches else f"未匹配到文件：{pattern}", True


def _grep(ctx: ToolContext, pattern: str, path: str = ".") -> tuple[str, bool]:
    """内容搜索：返回「文件:行号:内容」，支持正则。"""
    regex = re.compile(pattern)
    base = ctx.permissions.resolve_path(path)
    files = base.rglob("*") if base.is_dir() else [base]
    hits = []
    for f in files:
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{f.relative_to(ctx.root)}:{i}: {line}")
    return "\n".join(hits[:100]) if hits else f"未匹配到：{pattern}", True


def _bash(ctx: ToolContext, command: str) -> tuple[str, bool]:
    """执行 shell 命令（需 execute 档，且过黑名单；超时 30s）。"""
    if not ctx.permissions.tool_allowed("Bash"):
        return "错误：执行命令需 execute 权限档（默认关闭，请显式开启）", False
    ctx.permissions.check_command(command)          # 黑名单拦截（会抛 PermissionError_）
    try:
        proc = subprocess.run(command, shell=True, cwd=str(ctx.root),
                              capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（30s）", False
    out = (proc.stdout + proc.stderr).strip()
    return out[:4000] or "(无输出)", proc.returncode == 0


def _task(ctx: ToolContext, prompt: str) -> tuple[str, bool]:
    """派生隔离上下文的子 Agent（对应简历「Task 派生隔离上下文子 Agent」）。

    真正的子 Agent 派生逻辑在 router.py；这里只做转发。
    """
    if ctx.router is None:
        return "错误：子 Agent 路由未初始化", False
    return ctx.router.run_subagent(prompt), True


# ---------------------------------------------------------------------------
# 工具注册表：统一描述 + 分发
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    params: dict            # 参数名 -> 说明（用于 prompt 与校验）
    func: object


def build_registry() -> dict[str, Tool]:
    """构建 7 工具注册表。新增工具在此登记即可。"""
    return {
        "Read": Tool("Read", "读取文件内容（带行号）", {"path": "文件路径", "start": "起始行(可选)", "end": "结束行(可选)"}, _read),
        "Write": Tool("Write", "整文件写入/覆盖", {"path": "文件路径", "content": "完整内容"}, _write),
        "Edit": Tool("Edit", "精确替换：old_string 必须唯一命中", {"path": "文件路径", "old_string": "原文", "new_string": "替换文"}, _edit),
        "Glob": Tool("Glob", "按模式列文件", {"pattern": "glob 模式，如 **/*.py"}, _glob),
        "Grep": Tool("Grep", "正则搜索内容", {"pattern": "正则", "path": "目录或文件(默认.)"}, _grep),
        "Bash": Tool("Bash", "执行 shell 命令（需 execute 权限）", {"command": "命令"}, _bash),
        "Task": Tool("Task", "派生子 Agent 独立完成子任务", {"prompt": "子任务描述"}, _task),
    }


def describe_tools(registry: dict[str, Tool]) -> str:
    """把工具清单渲染成 prompt 片段（喂给主循环的 system prompt）。"""
    lines = []
    for t in registry.values():
        params = ", ".join(f"{k}: {v}" for k, v in t.params.items())
        lines.append(f"- {t.name}({params})：{t.description}")
    return "\n".join(lines)


def run_tool(registry: dict[str, Tool], name: str, args: dict,
             ctx: ToolContext) -> tuple[str, bool]:
    """统一分发入口：按工具名找到实现并执行。

    :return: (结果文本, 是否成功)
    权限/路径异常在这里统一捕获，转成可读的错误文本（不中断主循环）。
    """
    tool = registry.get(name)
    if tool is None:
        return f"错误：未知工具 {name}", False
    if not ctx.permissions.tool_allowed(name):
        return f"错误：工具 {name} 被权限档「{ctx.permissions.level}」拒绝", False
    try:
        return tool.func(ctx, **args)
    except PermissionError_ as e:
        return f"权限拦截：{e}", False
    except TypeError as e:
        return f"参数错误：{e}", False
    except Exception as e:  # 兜底：工具异常不能带崩主循环
        return f"工具执行异常：{type(e).__name__}: {e}", False
