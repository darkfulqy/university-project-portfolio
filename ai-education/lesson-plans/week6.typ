#import "template.typ": *

#title-block("Week 6", "用 Semgrep 扫描并修复安全漏洞", "2课时（90分钟）", "安全卫士")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "安全卫士",
      "使用 Semgrep 对代码进行静态安全分析，识别并修复常见安全漏洞，理解 SAST 与 SCA 的区别。",
      ("SAST", "Semgrep", "密钥泄露", "SQL注入", "XSS", "SCA")
    )

    #v(5pt)

    #info-card("项目目标", [
      - 理解 SAST 基本原理
      - 掌握 Semgrep 安装、配置与运行
      - 识别并修复常见安全漏洞
      - 区分有效告警与误报
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(5pt)

    #info-card("课前准备", [
      - 安装 Semgrep CLI
      - 确保 week6 项目完整
      - 预习 OWASP Top 10
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(5pt)

    #milestone("1", "安全扫描基础", "15分钟",
      "讲解 SAST/DAST/SCA 区别，介绍 Semgrep 规则引擎与规则市场",
      deliverable: "理解扫描类型")

    #v(5pt)

    #milestone("2", "运行 Semgrep 扫描", "15分钟",
      "执行 semgrep ci --subdir week6，分析后端 Python、前端 JS、依赖项和配置文件",
      deliverable: "扫描报告")

    #v(5pt)

    #milestone("3", "漏洞筛选与分析", "20分钟",
      "从报告中筛选 3 个真实安全问题，区分误报，理解风险等级与攻击面",
      deliverable: "漏洞分析记录")

    #v(5pt)

    #milestone("4", "修复实战", "30分钟",
      "用 AI 工具辅助修复：参数化 SQL、清理 DOM 写入、限制 CORS、升级依赖、移除硬编码密钥",
      deliverable: "修复后的代码")

    #v(5pt)

    #milestone("5", "验证与回归", "10分钟",
      "修复后重新跑 Semgrep 确认清除，运行全量测试确保功能正常",
      deliverable: "测试通过 + 对比记录")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "运行 Semgrep 扫描",
      "筛选 3 个真实安全问题",
      "记录忽略的误报及原因",
      "修复第1个漏洞",
      "修复第2个漏洞",
      "修复第3个漏洞",
      "重新扫描确认清除",
      "运行全量测试",
      "提交修复报告",
    ))

    #v(5pt)

    #student-zone("漏洞分析笔记", lines: 3)

    #v(5pt)

    #reflection-prompts((
      "你如何区分有效告警和误报？",
      "修复安全漏洞时，为什么要运行回归测试？",
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
      重点考察：Semgrep 扫描的完整性、漏洞分析的深度、修复方案的正确性、测试回归的严谨性，以及报告的清晰程度。
    ]
  ]
)
