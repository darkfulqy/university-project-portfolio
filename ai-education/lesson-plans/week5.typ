#import "template.typ": *

#title-block("Week 5", "用 Warp 进行智能体驱动开发", "2课时（90分钟）", "Warp自动化中心")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "Warp自动化中心",
      "在 Warp 终端中搭建 Drive 自动化和多智能体工作流，让多个 AI 同时在不同任务上协作。",
      ("Warp Drive", "Saved Prompts", "Rules", "Multi-Agent", "git worktree", "MCP")
    )

    #v(5pt)

    #info-card("项目目标", [
      - 熟悉 Warp 终端及智能体开发环境
      - 掌握 Warp Drive 的提示词/规则/MCP
      - 体验多智能体并发工作流
      - 学会用 git worktree 避免代码冲突
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(5pt)

    #info-card("课前准备", [
      - 注册并安装 Warp：warp.dev
      - 浏览 Warp 大学资源
      - 确保 week5 starter 可运行
      - 阅读 week5/docs/TASKS.md
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(5pt)

    #milestone("1", "Warp 环境概览", "15分钟",
      "介绍 Warp AI 功能、Drive 面板、多标签页智能体会话，对比传统终端",
      deliverable: "Warp配置完成")

    #v(5pt)

    #milestone("2", "Warp Drive 自动化 A", "25分钟",
      "创建 saved prompt / rule / MCP 服务器（至少一类），如测试运行器、文档同步、重构助手",
      deliverable: "Drive自动化配置")

    #v(5pt)

    #milestone("3", "多智能体工作流 B", "25分钟",
      "在多个 Warp 标签页中同时跑不同任务，体验并发协作，用 git worktree 隔离代码",
      deliverable: "多智能体运行中")

    #v(5pt)

    #milestone("4", "完成任务与验证", "20分钟",
      "从 TASKS.md 中选任务，用搭建的 Warp 自动化来完成，记录难度等级",
      deliverable: "任务完成记录")

    #v(5pt)

    #milestone("5", "总结与分享", "5分钟",
      "分享最多同时跑了几个智能体、踩了哪些坑、并发收益与风险",
      deliverable: "经验分享")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "创建 Warp Drive 自动化（≥1）",
      "设计多智能体工作流（≥1）",
      "用 git worktree 隔离代码",
      "完成 TASKS.md 中的任务",
      "记录每个任务的难度等级",
      "填写 writeup.md",
      "记录并发收益/风险/踩坑",
    ))

    #v(5pt)

    #student-zone("多智能体协调笔记", lines: 3)

    #v(5pt)

    #reflection-prompts((
      "同时运行多个智能体时，最大的挑战是什么？",
      "git worktree 在避免冲突方面起了什么作用？",
    ))
  ],

  ratios: (1fr, 1.3fr, 1fr),
  gutter: 12pt
)

#v(5pt)

#block(
  width: 100%,
  fill: c-bg-light,
  stroke: c-border,
  radius: 5pt,
  inset: 8pt,
  [
    #text(size: 10pt, weight: "bold", fill: c-secondary)["📊 评分标准"]
    #v(4pt)
    #text(size: 9pt)[
      开放性项目评分，重点考察：Warp 自动化的实用性、多智能体协调的设计合理性、writeup.md 的深度反思，以及严格在 week5/ 范围内修改的规范性。
    ]
  ]
)
