#import "template.typ": *

#title-block("Week 2", "行动项提取器", "2课时（90分钟）", "智能笔记助手")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "智能笔记助手",
      "在 FastAPI + SQLite 最小应用基础上，将规则驱动的行动项提取器升级为 LLM 驱动的智能助手。",
      ("FastAPI", "SQLite", "LLM驱动", "单元测试", "代码重构", "Agentic")
    )

    #v(8pt)

    #info-card("项目目标", [
      - 用 LLM 替换传统规则引擎
      - 编写覆盖多种场景的单元测试
      - 体验 AI 辅助编程（Cursor）
      - 理解 API 设计与重构最佳实践
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(8pt)

    #info-card("课前准备", [
      - 安装 Cursor IDE（教育版）
      - 激活 Conda：#raw("conda activate cs146s", lang: "bash")
      - 启动：#raw("poetry run uvicorn week2.app.main:app --reload", lang: "bash")
      - 访问 http://127.0.0.1:8000/
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(8pt)

    #milestone("1", "项目概览与Cursor配置", "15分钟",
      "介绍应用结构，演示用 Cursor 打开项目并运行",
      deliverable: "应用正常运行")

    #v(8pt)

    #milestone("2", "TODO 1：LLM驱动提取", "25分钟",
      "实现 extract_action_items_llm()，用 Ollama 输出结构化 JSON",
      deliverable: "extract.py + 测试通过")

    #v(8pt)

    #milestone("3", "TODO 2-3：测试与重构", "30分钟",
      "覆盖多种输入场景的单元测试；定义 API Schema、清理数据库层、加强错误处理",
      deliverable: "test_extract.py + 重构代码")

    #v(8pt)

    #milestone("4", "TODO 4：智能代理模式", "15分钟",
      "新增 LLM 提取接口端点和前端按钮；新增列出笔记接口",
      deliverable: "新增端点 + UI按钮")

    #v(8pt)

    #milestone("5", "TODO 5：生成README & 总结", "5分钟",
      "用 Cursor 分析代码库生成 README；回顾本周项目成果",
      deliverable: "README.md")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "TODO 1：LLM驱动提取",
      "TODO 2：单元测试",
      "TODO 3：重构代码",
      "TODO 4：Agentic Mode",
      "TODO 5：生成README",
      "填写 writeup.md",
      "代码注释标清改动",
    ))

    #v(8pt)

    #student-zone("Prompt记录", lines: 3)

    #v(8pt)

    #reflection-prompts((
      "LLM提取和规则提取相比，各有什么优劣？",
      "Cursor在哪些地方帮到了你？哪些地方还需要手动调整？",
    ))
  ],

  ratios: (1fr, 1.3fr, 1fr),
  gutter: 12pt
)

#v(8pt)

#rubric((
  "TODO 1 LLM提取：20分",
  "TODO 2 单元测试：20分",
  "TODO 3 重构：20分",
  "TODO 4 Agentic：20分",
  "TODO 5 README：20分",
  "代码10分+Prompt10分/项",
), total: "100分")
