# -*- coding: utf-8 -*-
"""agent 包：Claude Code 式终端 Coding Agent 的最小 MVP。

模块职责速览（面试按这张表讲）：
    loop        —— ReAct 主循环（解析→分发执行→结果回填→终止判定）
    llm         —— 本地模型封装 + 可离线演示的 Mock 模型
    tools       —— Read/Write/Edit/Glob/Grep/Bash/Task 七个核心工具
    permissions —— 三档权限 + 危险命令黑名单 + 路径越界检测
    context     —— token 计数 + 三级压缩（结果截断/历史摘要/会话重建）
    router      —— 三级路由（规则→向量→LLM）+ Task 子 Agent
    trajectory  —— 轨迹记录与评测（prompt/工具调用/diff/成功率）
"""
