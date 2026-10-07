#import "template.typ": *

#title-block("Week 8", "多技术栈 AI 加速网页应用开发", "2课时（90分钟）", "全栈应用三剑客")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "全栈应用三剑客",
      "用 3 种不同技术栈实现同一款功能完整的网页应用，理解不同技术栈的适用场景与 trade-offs。",
      ("Bolt.new", "MERN", "Django", "Next.js", "Rails", "CRUD")
    )

    #v(4pt)

    #info-card("项目目标", [
      - 用 3 种技术栈实现同一应用
      - 掌握 bolt.new AI 生成平台
      - 理解不同技术栈适用场景
      - 评估 AI 生成代码质量并修复
      - 体验概念到原型的快速迭代
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(4pt)

    #info-card("课前准备", [
      - 注册 Bolt（bolt.new）
      - 领取专属优惠码
      - 确定应用概念
      - 准备非 JS 技术栈环境
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(4pt)

    #milestone("1", "展示日与作业说明", "10分钟",
      "确认 Demo Day 安排，讲解最低功能要求：CRUD + 持久化 + 校验 + 可用 UI",
      deliverable: "概念确定")

    #v(4pt)

    #milestone("2", "版本1：Bolt生成", "25分钟",
      "用自然语言描述需求，让 bolt.new 生成第一版全栈应用，导出代码",
      deliverable: "v1-Bolt版本")

    #v(4pt)

    #milestone("3", "版本2：非JS技术栈", "25分钟",
      "选择 Django/Rails/Flask 等，手动或借助 AI 搭建第二版本",
      deliverable: "v2-非JS版本")

    #v(4pt)

    #milestone("4", "版本3：第三技术栈", "20分钟",
      "选择 MERN/MEVN/Next.js 等完成第三版本，确保功能等价",
      deliverable: "v3-第三版本")

    #v(4pt)

    #milestone("5", "文档与提交", "10分钟",
      "每个版本写 README（环境、安装、运行、环境变量），记录 AI 生成 vs 手动修复",
      deliverable: "3×README + writeup")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "确定应用概念",
      "版本1：Bolt.new 生成",
      "版本2：非 JS 技术栈",
      "版本3：第三技术栈",
      "每个版本写 README",
      "记录 AI生成 vs 手动修复",
      "填写 writeup.md",
      "准备 Demo Day",
    ))

    #v(4pt)

    #student-zone("技术栈对比笔记", lines: 3)

    #v(4pt)

    #reflection-prompts((
      "三个技术栈各有什么优缺点？适合什么场景？",
      "AI 生成代码后，你最常手动修复哪些问题？",
    ))
  ],

  ratios: (1fr, 1.3fr, 1fr),
  gutter: 12pt
)

#v(4pt)

#rubric((
  "应用概念：10分",
  "三种技术栈：10分",
  "使用 Bolt：10分",
  "非 JS 语言：10分",
  "版本1：20分",
  "版本2：20分",
  "版本3：20分",
), total: "100分")
