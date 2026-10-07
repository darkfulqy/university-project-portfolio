#import "template.typ": *

#title-block("Week 3", "搭建自定义 MCP 服务器", "2课时（90分钟）", "API封装大师")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "API封装大师",
      "设计并实现一个模型上下文协议（MCP）服务器，将真实外部API封装为AI可调用的工具。",
      ("MCP Protocol", "Tool定义", "错误处理", "Rate Limit", "STDIO/HTTP", "认证")
    )

    #v(4pt)

    #info-card("项目目标", [
      - 理解 MCP 核心概念：Tool/Resource/Prompt
      - 封装真实外部 API 为 MCP 服务
      - 定义带类型参数的工具与错误处理
      - 了解 STDIO 与 HTTP 传输场景
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(4pt)

    #info-card("课前准备", [
      - 阅读 MCP 官方文档
      - 选定外部 API（天气/GitHub/Notion/电影等）
      - 注册并获取 API 访问密钥
      - Python 3.12 + Conda
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(4pt)

    #milestone("1", "MCP协议概览", "15分钟",
      "讲解 MCP 诞生背景、核心能力，对比传统 function calling",
      deliverable: "理解Tool/Resource/Prompt")

    #v(4pt)

    #milestone("2", "选择API与工具设计", "15分钟",
      "学生确定外部 API，梳理端点，设计至少2个 MCP Tool 的 Schema",
      deliverable: "工具设计文档")

    #v(4pt)

    #milestone("3", "实现MCP服务器核心", "25分钟",
      "编写 server 代码，暴露 Tool，处理 HTTP 失败、超时、空结果、退避重试",
      deliverable: "server/ + main.py")

    #v(4pt)

    #milestone("4", "传输与部署", "20分钟",
      "本地 STDIO + Claude Desktop/Cursor；远程 HTTP + 可选 OAuth2/API Key",
      deliverable: "配置文件")

    #v(4pt)

    #milestone("5", "文档与调试", "15分钟",
      "编写 README（安装、配置、运行、客户端配置），用 inspector 调试",
      deliverable: "README.md")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "选定外部 API 并注册密钥",
      "设计 2+ 个 MCP Tool Schema",
      "实现 MCP Server 核心代码",
      "添加错误处理和重试逻辑",
      "本地部署测试（Claude/Cursor）",
      "编写 README 文档",
      "（可选）远程 HTTP 部署",
      "（可选）实现认证机制",
    ))

    #v(4pt)

    #student-zone("API选择与Schema设计", lines: 3)

    #v(4pt)

    #reflection-prompts((
      "MCP 和传统 API 调用相比，优势在哪里？",
      "你在处理错误和速率限制时遇到了什么挑战？",
    ))
  ],

  ratios: (1fr, 1.3fr, 1fr),
  gutter: 12pt
)

#v(4pt)

#rubric((
  "功能实现：35分",
  "可靠性：20分",
  "开发者体验：20分",
  "代码质量：15分",
  "远程HTTP：+5分",
  "认证：+5分",
), total: "100分")
