#import "template.typ": *

#title-block("Week 4", "现实中的自主编程智能体", "2课时（90分钟）", "开发者自动化工厂")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "开发者自动化工厂",
      "利用 Claude Code 的核心功能（斜杠命令、CLAUDE.md、子代理、MCP），搭建至少2个开发者自动化流程。",
      ("斜杠命令", "CLAUDE.md", "SubAgent", "MCP集成", "自动化", "开发者效率")
    )

    #v(5pt)

    #info-card("项目目标", [
      - 掌握 Claude Code 四种核心能力
      - 设计并搭建 2+ 开发者自动化流程
      - 用 AI 自动化提升测试/文档/重构效率
      - 体验多智能体协作工作模式
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(5pt)

    #info-card("课前准备", [
      - 安装 Claude Code CLI/桌面版
      - 阅读最佳实践文档
      - 阅读子代理文档
      - 克隆 week4 starter 项目
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(5pt)

    #milestone("1", "Claude Code 功能概览", "15分钟",
      "演示斜杠命令、CLAUDE.md 自动加载、子代理调用、MCP 工具接入",
      deliverable: "熟悉四种核心能力")

    #v(5pt)

    #milestone("2", "探索 Starter 应用", "10分钟",
      "运行 make run，浏览后端、前端、测试、pre-commit 配置",
      deliverable: "应用跑通")

    #v(5pt)

    #milestone("3", "搭建自动化 A", "25分钟",
      "从斜杠命令/CLAUDE.md/子代理/MCP中选一类，实现第一个自动化",
      deliverable: "自动化配置A")

    #v(5pt)

    #milestone("4", "搭建自动化 B", "25分钟",
      "选择另一类别，实现第二个自动化，建议跨类别混搭",
      deliverable: "自动化配置B")

    #v(5pt)

    #milestone("5", "让自动化干实事", "15分钟",
      "用已搭建的自动化改进或扩展 starter 应用，记录 Before vs After",
      deliverable: "应用改进 + 记录")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "搭建自动化流程 A",
      "搭建自动化流程 B",
      "用自动化改进 starter 应用",
      "填写 writeup.md",
      "记录设计灵感与参考文档",
      "记录运行命令与预期输出",
      "记录 Before vs After",
    ))

    #v(5pt)

    #student-zone("自动化设计笔记", lines: 3)

    #v(5pt)

    #reflection-prompts((
      "哪个自动化流程对你的开发效率提升最大？",
      "多智能体协作中，如何分配任务最合理？",
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
      开放性项目评分，重点考察：自动化流程的实用性、设计文档的完整性、对 Claude Code 功能的理解深度、以及实际增强 starter 应用的效果。
    ]
  ]
)
