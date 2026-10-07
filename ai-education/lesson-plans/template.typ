// 高中软件工程课程教案模板 - 横向A4 项目驱动版
#set page(
  paper: "a4",
  flipped: true,
  margin: (top: 10mm, bottom: 8mm, left: 10mm, right: 10mm),
  footer: [
    #set text(size: 8pt, fill: gray)
    #align(center)[
      现代软件开发者 (CS146S) · 高中生适用 · 项目驱动学习 · Stanford 2025 秋季课程本土化
    ]
  ]
)

#set text(font: ("Heiti SC", "Hiragino Kaku Gothic ProN", "PingFang SC"), size: 9pt)
#set heading(numbering: none)

// 定义颜色
#let c-primary = rgb("#2563eb")
#let c-secondary = rgb("#475569")
#let c-accent = rgb("#f59e0b")
#let c-success = rgb("#10b981")
#let c-bg-light = rgb("#f8fafc")
#let c-bg-warm = rgb("#fffbeb")
#let c-border = rgb("#e2e8f0")
#let c-student = rgb("#ecfdf5")
#let c-student-border = rgb("#a7f3d0")

// ========== 标题块 ==========
#let title-block(week, title, duration, project) = {
  block(
    width: 100%,
    fill: c-primary,
    radius: 6pt,
    inset: (x: 7pt, y: 5pt),
    [
      #set text(fill: white)
      #grid(
        columns: (auto, 1fr, auto),
        align: horizon,
        gutter: 8pt,
        [
          #block(fill: white, radius: 4pt, inset: (x: 8pt, y: 3pt))[
            #set text(fill: c-primary, weight: "bold", size: 10pt)
            #week
          ]
        ],
        [
          #text(size: 14pt, weight: "bold")[#title]
          #v(1.5pt)
          #text(size: 8.5pt, fill: rgb("#bfdbfe"))[📁 项目：#project]
        ],
        [
          #block(fill: rgb("#1d4ed8"), radius: 4pt, inset: (x: 8pt, y: 3pt))[
            #set text(fill: white, size: 8.5pt)
            ⏱ #duration
          ]
        ]
      )
    ]
  )
  v(6pt)
}

// ========== 信息卡片 ==========
#let info-card(title, content, icon: "📋", fill-color: c-bg-light, stroke-color: c-border) = {
  block(
    width: 100%,
    fill: fill-color,
    stroke: stroke-color,
    radius: 4pt,
    inset: 2.5pt,
    [
      #text(size: 9pt, weight: "bold", fill: c-secondary)[#icon #title]
      #v(2pt)
      #set text(size: 8.2pt)
      #content
    ]
  )
}

// ========== 标签 ==========
#let tag(text-content, color: c-primary) = {
  box(fill: color.lighten(85%), stroke: color.lighten(50%), radius: 3pt, inset: (x: 4pt, y: 1.2pt))[
    #set text(size: 7.2pt, fill: color.darken(30%))
    #text-content
  ]
}

// ========== 项目里程碑卡片 ==========
#let milestone(num, title, duration, desc, deliverable: none) = {
  block(
    width: 100%,
    fill: white,
    stroke: c-border,
    radius: 4pt,
    inset: 2.5pt,
    [
      #grid(
        columns: (20pt, 1fr),
        gutter: 4pt,
        align: top,
        [
          #circle(radius: 9pt, fill: c-accent, stroke: none)[
            #set text(fill: white, size: 8pt, weight: "bold")
            #align(center + horizon)[#num]
          ]
        ],
        [
          #grid(
            columns: (1fr, auto),
            gutter: 3pt,
            [
              #text(weight: "bold", size: 9pt)[#title]
            ],
            [
              #text(size: 7.8pt, fill: gray)[#duration]
            ]
          )
          #v(0.8pt)
          #text(size: 8pt, fill: c-secondary)[#desc]
          #if deliverable != none [
            #v(0.8pt)
            #box(fill: c-success.lighten(90%), stroke: c-success.lighten(60%), radius: 3pt, inset: (x: 4pt, y: 1pt))[
              #set text(size: 7.2pt, fill: c-success.darken(30%))
              📦 #deliverable
            ]
          ]
        ]
      )
    ]
  )
}

// ========== 学生工作区 / 笔记区 ==========
#let student-zone(title, lines: 3, icon: "✏️") = {
  block(
    width: 100%,
    fill: c-student,
    stroke: c-student-border,
    radius: 4pt,
    inset: 2.5pt,
    [
      #text(size: 9pt, weight: "bold", fill: c-success.darken(20%))[#icon #title]
      #v(2pt)
      #for i in range(lines) [
        #line(length: 100%, stroke: (dash: "dotted", paint: c-student-border, thickness: 1pt))
        #v(12pt)
      ]
    ]
  )
}

// ========== 任务清单区 ==========
#let task-checklist(items) = {
  block(
    width: 100%,
    fill: c-bg-warm,
    stroke: rgb("#fcd34d").lighten(60%),
    radius: 4pt,
    inset: 2.5pt,
    [
      #text(size: 9pt, weight: "bold", fill: rgb("#b45309"))["📋 我的任务清单"]
      #v(2pt)
      #for item in items [
        #grid(
          columns: (9pt, 1fr),
          gutter: 4pt,
          align: top,
          [
            #box(width: 9pt, height: 9pt, stroke: c-secondary, radius: 2pt)[#align(center + horizon)[#text(size: 6pt)[ ]]]
          ],
          [
            #text(size: 7.9pt)[#item]
          ]
        )
        #v(2pt)
      ]
    ]
  )
}

// ========== 反思区 ==========
#let reflection-prompts(prompts) = {
  block(
    width: 100%,
    fill: rgb("#eff6ff"),
    stroke: rgb("#bfdbfe"),
    radius: 4pt,
    inset: 2.5pt,
    [
      #text(size: 9pt, weight: "bold", fill: c-primary)["💭 课后反思"]
      #v(2pt)
      #for (i, prompt) in prompts.enumerate() [
        #text(size: 8pt, weight: "bold", fill: c-secondary)[Q#(i+1). #prompt]
        #v(1pt)
        #line(length: 100%, stroke: (dash: "dotted", paint: rgb("#bfdbfe"), thickness: 1pt))
        #v(5pt)
        #line(length: 100%, stroke: (dash: "dotted", paint: rgb("#bfdbfe"), thickness: 1pt))
        #v(7pt)
      ]
    ]
  )
}

// ========== 评分标准 ==========
#let rubric(items, total: none) = {
  block(
    width: 100%,
    fill: c-bg-light,
    stroke: c-border,
    radius: 4pt,
    inset: 2.5pt,
    [
      #text(size: 9pt, weight: "bold", fill: c-secondary)["📊 评分标准"]
      #v(2pt)
      #grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 4pt,
        ..items.map(item => text(size: 8pt)[#item])
      )
      #if total != none [
        #v(2pt)
        #align(right)[
          #box(fill: c-primary, radius: 4pt, inset: (x: 6pt, y: 2pt))[
            #set text(fill: white, size: 8pt, weight: "bold")
            满分：#total
          ]
        ]
      ]
    ]
  )
}

// ========== 项目背景卡 ==========
#let project-card(name, description, skills) = {
  block(
    width: 100%,
    fill: white,
    stroke: c-primary.lighten(60%),
    radius: 4pt,
    inset: 2.5pt,
    [
      #text(size: 9.5pt, weight: "bold", fill: c-primary)["🚀 本周项目：#name"]
      #v(1.5pt)
      #text(size: 8pt, fill: c-secondary)[#description]
      #v(2pt)
      #for skill in skills [
        #tag(skill, color: c-primary)
        #h(2pt)
      ]
    ]
  )
}

// ========== 两栏/三栏布局 ==========
#let two-col(left-content, right-content, left-ratio: 1fr, right-ratio: 1fr, gutter: 12pt) = {
  grid(
    columns: (left-ratio, right-ratio),
    gutter: gutter,
    left-content,
    right-content
  )
}

#let three-col(a, b, c, ratios: (1fr, 1fr, 1fr), gutter: 12pt) = {
  grid(
    columns: ratios,
    gutter: gutter,
    a, b, c
  )
}
