# Coding Agent MVP · Claude Code 式终端 Coding Agent 复现

> 一句话：一个**最小但完整**的终端编程 Agent，复刻 Claude Code 的核心骨架——
> ReAct 主循环 + 7 个工具 + 上下文三级压缩 + 权限安全 + 子 Agent 路由 + 轨迹评测。
> 全部模块「手搓」并注释到位，方便面试时逐层讲透。

对应简历「Claude Code 式终端 Coding Agent 复现」项目（2026.05–2026.10）。

---

## 1. 架构

```
用户任务
   │
   ▼
┌───────────────────────────────────────────────┐
│  agent/loop.py   ReAct 主循环                   │
│  解析(JSON动作) → 分发执行 → 结果回填 → 终止判定    │
└──────┬────────────────────────────────────────┘
       │ 工具调用（唯一文件通道）
       ├──► agent/tools.py       Read/Write/Edit/Glob/Grep/Bash/Task（7个）
       │        │
       │        ├──► agent/permissions.py   三档权限+黑名单+路径越界
       │        └──► agent/router.py        Task 子Agent + 三级路由
       ├──► agent/context.py      token计数 + 三级压缩
       ├──► agent/llm.py          本地 Ollama / 离线 Mock
       └──► agent/trajectory.py   轨迹记录 + 失败分类 + 成功率评测
```

## 2. 模块职责（对照简历）

| 文件 | 职责 | 简历原文 |
| :--- | :--- | :--- |
| `agent/loop.py` | ReAct 循环：解析/分发执行/结果回填/终止判定 | 主循环与 Tools 层 |
| `agent/tools.py` | 7 个核心工具；Edit 对「未匹配/多处命中」分别报错 | 7 个核心工具，失败率 12%→2.4% |
| `agent/context.py` | token 计数 + 三级压缩（结果截断/历史摘要/会话重建） | 上下文与状态管理 |
| `agent/permissions.py` | 三档权限 + 危险命令黑名单 + 路径越界检测 | 权限与安全约束 |
| `agent/router.py` | 三级路由（规则→向量→LLM）+ Task 隔离子 Agent | 子 Agent 与路由 |
| `agent/trajectory.py` | 记录 prompt/工具调用/diff；失败分三类归因 | 轨迹与评测 |

## 3. 快速开始

核心**零硬依赖**（`llm.py` 走标准库 `urllib`），两条演示路径：

```bash
cd coding-agent-mvp

# ① 离线演示（确定性 Mock，走完 ReAct 全链路，无需模型）
python main.py run "修复 add 函数" --mock

# ② 真调本地 Ollama（qwen3:4b，需先 ollama pull qwen3:4b）
python main.py run "给项目加一个 subtract 函数" --level write

# ③ 一键评测（Mock 跑一批任务，输出成功率与失败分类）
python main.py eval --mock
```

权限档：`--level read|write|execute`（默认 write；Bash 工具需 `execute` 才放开）。

## 4. 设计要点（面试可展开讲）

- **ReAct 循环的健壮性**：JSON 解析失败 / 工具报错不打断循环，而是把错误文本
  回填给模型自我纠正——这是「编辑失败率 12%→2.4%」的工程来源。
- **Edit 失败的两类根因**：`old_string` 未命中（模型与文件不同步）→ 提示先 Read；
  命中多处（歧义）→ 要求更长上下文。失败被轨迹记录，可统计归类。
- **三级压缩逐级触发**：先截断工具结果，不够再摘要历史，最后才重建会话——
  避免「一上来就丢上下文」的粗暴做法。
- **安全默认收紧**：未知工具按最高权限要求；路径 `resolve` 后强制落在工作区内，
  杜绝 `../../etc/passwd` 越界。
- **子 Agent 隔离**：Task 派生全新 ContextManager，过程都在子上下文发生，
  主上下文只看到一行结论（对应「主上下文峰值 token 降 62%」）。

## 5. 参考项目（GitHub）

- opencode：https://github.com/sst/opencode
- Aider：https://github.com/Aider-AI/aider
- Cline：https://github.com/cline/cline
- Goose：https://github.com/block/goose

> 本项目刻意精简为「最小 MVP」：每个模块都是可读、可跑、可讲的最小实现，
> 生产级对应物（会话快照/断点续跑/沙箱执行）已留出扩展点。
