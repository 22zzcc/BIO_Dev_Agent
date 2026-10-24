#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py —— 基于生物制药的建模 Agent 入口
=========================================

围绕三要素（工作流 / 组件化 / 记忆）装配 agent：

    工作流     workflow.router（意图路由）+ workflow.loop（编排执行 + 轨迹）
    组件化     components.modeling（蛋白建模）+ components.rag（agentic RAG）+ feedback
    记忆       memory.context（短期）+ memory.longterm（长期）+ memory.critic（评论家）

用法（在项目根目录）：

    # 一句话请求，自动路由到「建模 / 诊断 / 问答」
    python main.py run "分析 9MW2821 抗体的演化表征与药性特性"
    python main.py run "抗体 SEC-HPLC 纯度偏低有哪些原因？"
    python main.py run "9MW2821 的Ⅲ期临床主要终点是什么？"
    python main.py run "计算序列 EVQLVESGGGLVQPGGSLRLSCAAS 的分子量和等电点"   # 显式序列→建模组件

    # 记忆闭环演示（评论家打分 + badcase 落库 + 偏好对反哺）
    python main.py feedback
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_agent(args):
    from agent.llm import OllamaLLM
    from agent.components.base import Registry
    from agent.components.modeling import build_modeling_component
    from agent.components.rag import (RAGInterface, build_rag_component,
                                      build_feedback_component)
    from agent.workflow.router import Router
    from agent.workflow.loop import Workflow
    from agent.memory.longterm import LongTermMemory
    from agent.memory.critic import Critic

    # 组件层：建模 + RAG + 反馈 三个可插拔组件
    rag = RAGInterface(backend=args.rag_backend, use_llm=False)
    registry = Registry()
    registry.register(build_modeling_component())
    registry.register(build_rag_component(rag))
    registry.register(build_feedback_component(rag))

    # 工作流层：路由 + 编排
    llm = None if args.rule_only else OllamaLLM()   # LLM 只作兜底，规则路由离线可跑
    router = Router(llm=llm)
    workflow = Workflow(registry, router)

    # 记忆层：长期 + 评论家
    memory = LongTermMemory()
    critic = Critic(memory, llm=llm)
    return workflow, critic, rag


def cmd_run(args) -> None:
    workflow, critic, rag = build_agent(args)
    final, traj = workflow.run(args.request)

    print("=" * 70)
    print(f"请求：{args.request}")
    print("-" * 70)
    print("分析结果：")
    print(final)
    print("-" * 70)
    print("轨迹（工作流 → 组件）：")
    for i, s in enumerate(traj.steps, start=1):
        print(f"  {i}. [{s['component']}:{s['action']}] "
              f"{'✓' if s['ok'] else '✗'}  {s['thought']}")


def cmd_feedback(args) -> None:
    """记忆闭环演示：评论家打分 → badcase 落库 → 偏好对反哺。"""
    workflow, critic, rag = build_agent(args)

    print("=" * 70)
    print("① 评论家打分：一个「零召回」坏例")
    q = "9MW2821 抗体的结合亲和力是多少？"
    verdict = critic.judge(q, "资料中未提及", citations=[], hit_count=0)
    print(f"   得分 {verdict['score']}，issues={verdict['issues']}")

    print("② badcase 落库长期记忆")
    critic.collect(q, "资料中未提及", citations=[], hit_count=0)
    print(f"   记忆 {critic.memory.stats()}")

    print("③ 用户纠正 → 偏好对（喂 PPO/DPO）")
    critic.collect_feedback("9MW2821 靶向什么靶点？", "靶向 HER2", "靶向 Nectin-4（更准确）")
    fb = critic.feedback()
    print(f"   偏好对 {len(fb['preference_pairs'])} 条：")
    for p in fb["preference_pairs"]:
        print(f"     prompt  : {p['prompt']}")
        print(f"     rejected: {p['rejected']}")
        print(f"     chosen  : {p['chosen']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="基于生物制药的建模 Agent（工作流/组件化/记忆）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="一句话请求（自动路由到建模/诊断/问答）")
    p_run.add_argument("request", help="自然语言请求")
    p_run.add_argument("--rag-backend", default="local", choices=["local", "remote"])
    p_run.add_argument("--rule-only", action="store_true",
                       help="只用规则路由，不加载 LLM（离线）")
    p_run.set_defaults(func=cmd_run)

    p_fb = sub.add_parser("feedback", help="记忆闭环演示")
    p_fb.add_argument("--rag-backend", default="local", choices=["local", "remote"])
    p_fb.add_argument("--rule-only", action="store_true")
    p_fb.set_defaults(func=cmd_feedback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
