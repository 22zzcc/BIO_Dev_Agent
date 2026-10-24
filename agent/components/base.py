# -*- coding: utf-8 -*-
"""
components/base.py —— 组件化：Component 统一接口 + 注册表
===========================================================

「组件化」的落点：所有能力（建模、RAG、未来可加的药物警戒、临床分析…）都实现为
同一个 Component 接口，工作流层只认这个接口，与具体实现解耦。

一个组件 = 名字 + 描述 + 参数说明 + 一个可调用函数。
新增组件只需：实现一个 `(ctx, **args) -> (结果文本, 是否成功)` 的函数并注册，
工作流与路由都不用改——这就是「组件化」带来的可扩展性。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Component:
    """一个可插拔组件。func 签名统一为 (ctx, **args) -> (str, bool)。"""
    name: str
    description: str
    params: dict            # 参数名 -> 说明
    func: object

    def run(self, ctx, **args):
        """执行组件，异常不抛出、转成错误文本（组件异常不能带崩工作流）。"""
        try:
            return self.func(ctx, **args)
        except TypeError as e:
            return f"参数错误：{e}", False
        except Exception as e:
            return f"组件执行异常：{type(e).__name__}: {e}", False


class Registry:
    """组件注册表：注册/运行/描述，供工作流与路由使用。"""

    def __init__(self):
        self.components: dict[str, Component] = {}

    def register(self, comp: Component) -> None:
        self.components[comp.name] = comp

    def run(self, name: str, ctx, **args):
        """按组件名分发执行。"""
        comp = self.components.get(name)
        if comp is None:
            return f"错误：未知组件 {name}", False
        return comp.run(ctx, **args)

    def describe(self) -> str:
        lines = []
        for c in self.components.values():
            params = ", ".join(f"{k}: {v}" for k, v in c.params.items())
            lines.append(f"- {c.name}({params})：{c.description}")
        return "\n".join(lines)
