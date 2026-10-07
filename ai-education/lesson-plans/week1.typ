#import "template.typ": *

#title-block("Week 1", "提示词技巧", "2课时（90分钟）", "AI提示词大师")

#three-col(
  // ===== 左栏：项目信息 =====
  [
    #project-card(
      "AI提示词大师",
      "通过设计精准提示词，让本地大语言模型完成6种不同任务。你将像训练助手一样，学会与AI高效协作。",
      ("K-shot", "思维链", "工具调用", "自一致性", "RAG", "反思")
    )

    #v(8pt)

    #info-card("项目目标", [
      - 掌握6种核心提示词技巧
      - 在本地运行开源大模型（Ollama）
      - 培养与AI协作解决问题的思维
    ], icon: "🎯", fill-color: rgb("#eff6ff"), stroke-color: rgb("#bfdbfe"))

    #v(8pt)

    #info-card("课前准备", [
      - 安装 Ollama
      - 拉取模型：
        #raw("ollama run mistral-nemo:12b", lang: "bash")
        #raw("ollama run llama3.1:8b", lang: "bash")
      - Python 3.12 + Conda
    ], icon: "🛠")
  ],

  // ===== 中栏：项目里程碑 =====
  [
    #text(size: 11pt, weight: "bold", fill: c-secondary)["项目阶段"]
    #v(8pt)

    #milestone("1", "环境搭建", "15分钟",
      "演示本地运行开源大模型，对比本地部署 vs 云端调用的优劣",
      deliverable: "运行Ollama")

    #v(8pt)

    #milestone("2", "K-shot & 思维链", "25分钟",
      "通过少量示例引导模型理解任务格式；教模型一步步拆解推理过程",
      deliverable: "k_shot + chain_of_thought")

    #v(8pt)

    #milestone("3", "工具调用 & 自一致性", "25分钟",
      "让模型学会使用外部工具扩展能力；多次采样投票提升答案可靠性",
      deliverable: "tool_calling + self_consistency")

    #v(8pt)

    #milestone("4", "RAG & 反思", "20分钟",
      "检索增强让模型基于文档作答；让模型自我评估并迭代改进",
      deliverable: "rag + reflexion")

    #v(8pt)

    #milestone("5", "总结与练习", "5分钟",
      "回顾6种技巧的核心思想，学生开始动手完成代码练习",
      deliverable: "开始课后作业")
  ],

  // ===== 右栏：学生工作区 =====
  [
    #task-checklist((
      "完成 k_shot_prompting.py",
      "完成 chain_of_thought.py",
      "完成 tool_calling.py",
      "完成 self_consistency_prompting.py",
      "完成 rag.py",
      "完成 reflexion.py",
      "保存最终提示词和输出",
    ))

    #v(8pt)

    #student-zone("课堂笔记", lines: 3)

    #v(8pt)

    #reflection-prompts((
      "哪种提示词技巧对你的任务最有效？为什么？",
      "在本地运行大模型和调用云端API各有什么优缺点？",
    ))
  ],

  ratios: (1fr, 1.3fr, 1fr),
  gutter: 12pt
)

#v(8pt)

#rubric((
  "K-shot 提示：10分",
  "思维链：10分",
  "工具调用：10分",
  "自一致性：10分",
  "RAG：10分",
  "反思：10分",
), total: "60分")
