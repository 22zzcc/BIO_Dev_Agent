# Bio Modeling Agent · 基于生物制药的建模 Agent

> 一句话：一个围绕**三要素**（工作流 / 组件化 / 记忆）编排的生物制药建模 Agent。
> 建模取代 coding——生物制药公司不缺「会写代码的 agent」，缺「会做蛋白分子建模、
> 演化表征与药性推导的 agent」。

从我的 Coding Agent 项目重构而来：**同一套 harness，换工具层**，并围绕三要素重新编排。

```
请求
 │
 ▼
┌─ ① 工作流 workflow ──────────────────────────────────────┐
│   router.py  意图路由：建模 / 诊断 / 问答 / 显式序列 → 组件+动作  │
│   loop.py    按计划编排执行各组件，记录轨迹                     │
└───────────────┬──────────────────────────────────────────┘
                │ 调度（统一接口 base.Component）
┌───────────────▼──────────────────────────────────────────┐
│ ② 组件化 components（可插拔，与工作流解耦）                     │
│   modeling   蛋白建模：理化性质 / 演化保守性（纯计算，零 RAG 依赖）  │
│   rag        agentic RAG：问答 / 多因诊断 / 药性推导（组件接入）   │
│   feedback   用户反馈落库                                      │
└───────────────┬──────────────────────────────────────────┘
                │ 读写状态
┌───────────────▼──────────────────────────────────────────┐
│ ③ 记忆 memory                                              │
│   context   短期记忆：token 计数 + 三级压缩                      │
│   longterm  长期记忆：badcase / 反馈落库（跨会话沉淀）              │
│   critic    评论家：打分 + badcase 收取 + 反哺偏好对             │
└──────────────────────────────────────────────────────────┘
```

## 三要素

| 要素 | 目录 | 职责 |
| :--- | :--- | :--- |
| **工作流** | `agent/workflow/` | 意图路由（规则优先、LLM 兜底）+ 编排执行 + 轨迹 |
| **组件化** | `agent/components/` | 统一 `Component` 接口，建模/RAG/反馈可插拔 |
| **记忆** | `agent/memory/` | 短期上下文 + 长期反馈 + 评论家（打分/badcase/反哺） |

## 快速开始

核心零硬依赖；「建模组件」纯 Python 离线秒级，RAG 组件依赖本地 `mabwell-agentic-rag`。

```bash
cd coding-agent-mvp

# 一句话请求，自动路由到「建模 / 诊断 / 问答」（--rule-only 离线，无需 LLM）
python main.py run "分析 9MW2821 抗体的演化表征与药性特性" --rule-only
python main.py run "抗体 SEC-HPLC 纯度偏低有哪些原因？" --rule-only
python main.py run "9MW2821 的Ⅲ期临床主要终点是什么？" --rule-only

# 显式序列 → 建模组件（纯计算，不依赖 RAG）
python main.py run "计算序列 EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMH 的分子量和等电点" --rule-only

# 记忆闭环（评论家打分 + badcase 落库 + 偏好对反哺）
python main.py feedback --rule-only
```

> RAG 组件后端：`--rag-backend local`（直连本地 mabwell 项目）或 `remote`（走其 FastAPI）。
> 「agentic RAG 不丢，作为 bio 的一个组件，通过 `components/rag.py` 的接口接入」。

## 设计要点（面试可讲）

- **「建模」取代「coding」**：工具从 Read/Write/Bash 换成 `modeling`(蛋白建模) + `rag`
  (知识推导) + `feedback`(反馈)。同一套工作流/记忆骨架，换组件层即换领域。
- **组件化 = 可扩展**：新增组件只需实现 `(ctx, **args) -> (str, bool)` 并注册，
  工作流与路由零改动（未来可加药物警戒、临床分析等组件）。
- **工作流 = 路由与执行解耦**：router 决定「做什么」（规则优先，LLM 兜底），
  loop 负责「怎么做」（编排 + 轨迹），组件异常不中断。
- **记忆 = 短期 + 长期 + 评论家**：短期管 token 预算，长期沉淀 badcase/反馈，
  评论家像 PPO 的 critic 一样打分、但多一个 badcase 收取，形成越用越准的自我迭代。

## 参考项目（GitHub）

- opencode：https://github.com/sst/opencode
- Aider：https://github.com/Aider-AI/aider
- Cline：https://github.com/cline/cline
- gpt-researcher：https://github.com/assafelovic/gpt-researcher
