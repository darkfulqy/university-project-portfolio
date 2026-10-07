#import "template.typ": *

#title-block("Week 7", "用 Graphite 体验 AI 代码审查", "2课时（90分钟）", "代码审查官")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "代码审查官",
      "在 Graphite 平台上完成基于分支的开发流程，创建高质量 PR，对比人工审查与 AI 审查的优劣。",
      ("Git Branching", "Pull Request", "Code Review", "Graphite", "AI Review", "反思")
    )

    #v(5pt)

    #info-card("项目目标", [
      - 掌握基于分支的开发流程
      - 学会创建高质量 Pull Request
      - 体验 Graphite Diamond AI 辅助审查
      - 对比人工审查与 AI 审查优劣
      - 形成对 AI 审查可信度的判断
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(5pt)

    #info-card("课前准备", [
      - 注册 Graphite（教育码 CS146S）
      - 确保 week7 项目可运行
      - 阅读 week7/docs/TASKS.md
      - 准备 AI 编程工具
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(5pt)

    #milestone("1", "Graphite 工作流介绍", "15分钟",
      "注册与配置 Graphite，理解基于栈的分支管理 vs 传统 Git Flow",
      deliverable: "Graphite配置完成")

    #v(5pt)

    #milestone("2", "任务 1 实战", "20分钟",
      "创建独立分支 → AI 工具 1-shot 完成 → 手动逐行审查 → 提交 PR",
      deliverable: "PR #1")

    #v(5pt)

    #milestone("3", "AI 审查体验", "15分钟",
      "用 Graphite Diamond 生成 AI 代码审查，对比人工发现 vs AI 发现的问题",
      deliverable: "AI审查记录")

    #v(5pt)

    #milestone("4", "任务 2-4 并行推进", "30分钟",
      "重复分支 → AI 实现 → 人工审查 → PR → AI 审查，完成剩余任务",
      deliverable: "PR #2-4")

    #v(5pt)

    #milestone("5", "反思与总结", "10分钟",
      "对比4个 PR 的人工与 AI 审查结果，讨论 AI 审查强弱项与适用场景",
      deliverable: "反思总结")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "注册配置 Graphite",
      "任务1：分支+AI实现+审查+PR",
      "任务2：分支+AI实现+审查+PR",
      "任务3：分支+AI实现+审查+PR",
      "任务4：分支+AI实现+审查+PR",
      "生成 Graphite AI 审查",
      "填写 writeup.md",
    ))

    #v(5pt)

    #student-zone("人工审查记录", lines: 3)

    #v(5pt)

    #reflection-prompts((
      "人工审查时你关注哪些方面？",
      "AI 审查在哪些方面比人强？哪些方面比人弱？",
    ))
  ],

  ratios: (1fr, 1.3fr, 1fr),
  gutter: 12pt
)

#v(5pt)

#rubric((
  "任务1：20分",
  "任务2：20分",
  "任务3：20分",
  "任务4：20分",
  "反思：20分",
  "审查深度+AI审查",
), total: "100分")
