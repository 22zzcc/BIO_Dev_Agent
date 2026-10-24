# -*- coding: utf-8 -*-
"""
agent/permissions.py —— 三档权限 + 危险命令黑名单 + 路径越界检测
====================================================================

对应简历「三档权限叠加危险命令黑名单与路径越界检测，300 条指令拦截率 96%、
误杀率 3%」。这是 Coding Agent 敢在真实仓库上跑的前提：Agent 是「会执行代码的
代码」，必须给它套上**最小权限**的笼子。

三档权限（由低到高）：
    read    —— 只能读（Read/Glob/Grep）
    write   —— 可读写改（+Write/Edit），默认档
    execute —— 可执行命令（+Bash），需显式开启

两层安全校验：
    1. 命令黑名单：正则匹配危险命令（rm -rf / sudo / 管道给 sh 等）；
    2. 路径越界：把工具要读写的路径 resolve 后，强制落在 workspace 根内，
       杜绝 `../../etc/passwd` 这类越界读写。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# 工具 → 所需最低权限档
TOOL_LEVEL = {
    "Read": "read", "Glob": "read", "Grep": "read",
    "Write": "write", "Edit": "write",
    "Bash": "execute",
    "Task": "read",          # 子 Agent 的权限由其自身独立判定
}
_LEVEL_RANK = {"read": 0, "write": 1, "execute": 2}

# 危险命令黑名单（正则，命中即拒绝）。覆盖「删库/提权/反弹 shell/下载即执行」等
DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\b",      # rm -rf / rm -fr
    r"\bsudo\b",                                          # 提权
    r"\bmkfs\b", r"\bdd\s+if=",                           # 格式化 / 写盘
    r"\b(curl|wget)\b.*\|\s*(ba)?sh\b",                   # 下载即执行
    r">\s*/dev/(sd|nvme|disk)",                           # 直接写块设备
    r"\bchmod\s+777\b", r"\bchown\s+-R\b",                # 大开权限 / 递归改属主
    r"\bshutdown\b", r"\breboot\b", r"\b:\(\)\s*\{",       # 关机/重启/fork 炸弹
]


@dataclass
class PermissionError_(Exception):
    """权限拒绝异常（带原因，方便轨迹里记录「为什么被拦」）。"""
    reason: str

    def __str__(self):
        return self.reason


class Permissions:
    def __init__(self, level: str = "write", workspace_root: str | Path = "."):
        self.level = level
        self.root = Path(workspace_root).resolve()
        self.denied_count = 0      # 被拦截次数（简历「拦截率」的数据来源）

    def tool_allowed(self, tool_name: str) -> bool:
        """工具所需档位是否 <= 当前档位。"""
        need = TOOL_LEVEL.get(tool_name, "execute")   # 未知工具一律按最高要求，宁严勿松
        return _LEVEL_RANK[need] <= _LEVEL_RANK[self.level]

    def check_command(self, command: str) -> None:
        """命令黑名单校验；命中则抛 PermissionError_。"""
        for pat in DANGEROUS_PATTERNS:
            if re.search(pat, command, flags=re.I):
                self.denied_count += 1
                raise PermissionError_(f"命令命中危险黑名单：{command!r}")

    def resolve_path(self, path: str | Path) -> Path:
        """路径越界检测：把相对/绝对路径解析后，强制约束在 workspace 根内。

        - 相对路径 → 以 workspace 根为基准解析；
        - 解析后不在根内（如 ../.. 逃逸）→ 拒绝。
        返回「被约束后的绝对路径」或抛 PermissionError_。
        """
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        resolved = p.resolve()
        # 判断是否在根内：用 os.path.commonpath 兼容性最好
        try:
            resolved.relative_to(self.root)
        except ValueError:
            self.denied_count += 1
            raise PermissionError_(f"路径越界被拦截：{path!r} 逃出了工作区 {self.root}")
        return resolved
