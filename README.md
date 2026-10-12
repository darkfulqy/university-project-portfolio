# University Project Portfolio

李卿阳 · 上海大学计算机科学与技术本科 · 2024 年入学，预计 2028 年毕业

从计算机视觉数据准备与文保资料数字化，到 RAG 和 Agent 系统开发，再到 EEG 适配、泛化与数据高效学习研究。这里汇集项目经历、科研探索、竞赛和个人工程，供申请材料审阅；学校课程作业集中在独立栏目。

[33 条完整经历](docs/experience-catalog.md) · [成长时间线](docs/timeline.md) · [Claude 报告逐项对应](docs/report-crosswalk.md)

## 项目经历

| 项目 | 工作与内容 | 材料入口 |
| --- | --- | --- |
| [麦积山数字人文平台](maijishan-platform/) | 本科生大作业项目（约 30 人团队）；核心成员、6 人小组组长，负责组织协调、资料治理、混合检索与网站开发 | FastAPI、React、检索及管理源码 |
| [基于 YOLO 的数字散斑相关法物体形变测量](yolo-speckle-detection/) | 大学生创新创业训练项目技术负责人；散斑数据集与标注、YOLOv8-OBB 识别形变/位移/旋转、角点拟合位移网格、Ncorr 计算形变参数 | 项目与个人工作说明；拟合计算代码见课程作业栏目 |
| [司法文书 RAG 项目](legal-rag-prototype/) | 新致杯 AI 创新大赛 3 人团队组长；RAG 后端、知识库与检索架构，进入决赛并获优胜奖 | 历史原型源码 |
| [采购数据治理与智采通](procurement-data-governance/) | 由采购课程工程延伸的团队项目与校内创新竞赛；数据治理及系统组织 | 公告采集、原测试与知识图谱规范 |

散斑形变测量中的最小二乘拟合环节另以 C++ 实现，并作为[计算思维课程作业](coursework/grid-polynomial-fitting/)提交。

## 科研探索

| 方向 | 可查看的材料 | 当前材料所支持的阶段 |
| --- | --- | --- |
| [EEG 测试时适配与方法选择](eeg-tta-research/) | 跨会话基线、选择器和统计分析选段 | 独立研究；Workshop 拒稿后修订，未发表 |
| [EEG 数据蒸馏](eeg-distillation/) | EEG-DLite 复现驱动、LGM 适配和指标检查 | 对照实验与方法复核，保留上游归属 |
| [EEG 电极拓扑泛化](eeg-topology/) | 测量几何、参考变换与可辨识性实验 | 阶段性研究，限定模板与合成实验的结论 |
| [Fly-to-EEG](fly-to-eeg/) | 连接组仿真、动力学和前向观测原型 | 数值先导，人类 EEG 迁移尚未完成 |
| [VLA 低延迟探索](vla-latency-research/) | Phase-0 评测包装器与计时诊断 | 从阅读进入工程准备，未宣称加速成功 |

[科研工作流与阅读](research-workflow/)记录假设、基线、证据复核与失败分支。文保方向的[云冈石窟佛头论文合作](docs/experience-catalog.md#heritage-paper)按成长报告记录为第四作者、返修审稿中，不列为已发表论文。

## 竞赛与个人工程

| 项目 | 内容 |
| --- | --- |
| [数模国赛 A 题](math-modeling-2026/) | 团队水热耦合模型、Q1—Q4 求解器及历史数值检查 |
| [AI 股票信息研究](ai-stock-research/) | 命令行、来源适配、证据追踪和测试 |
| [行情提醒器模块](market-alerts/) | 数据类型、合成行情、指标与原测试；精选子集不含完整看板 |
| [RD-Agent 加密因子扩展](rd-agent-crypto-extension/) | 基于 Microsoft RD-Agent 的个人场景、评价和循环扩展 |
| [AI 教育课程设计](ai-education/) | 15 课方案与另一阶段的 16 课时教案源文件 |

报告中的[浏览器 Agent](docs/experience-catalog.md#browser-agent)也保留了经历说明；尚未定位到与其明确对应的独立工程时，不以其他项目代码代替。[文件 RAG 服务](docs/experience-catalog.md#file-rag)是长亮科技实习期间为公司开发的工具，归入实习经历。

## 实习与社团

- [长亮科技 AI Agent 与 RAG 实习](internship-agent-rag/)：2026 年 7—8 月，工作主题与个人职责摘要。
- [郁金香投资俱乐部与教学链方案](club-blockchain/)：社团组织事务、技术选型与建设规划。

## 课程作业与基础练习

[进入课程作业栏目](coursework/)：数字逻辑、OpenMP/CUDA、LINPACK 课程线索、C++ 多项式拟合、排序算法、数据结构、8086 汇编，以及单独标记的基础学习练习。

## 材料范围

本次在原仓库基础上续写，保留原项目内容并按学习与课外经历重新编排。经历清单依据 2026-10-03 Claude 整合报告与本地材料核对，整理截至 2026-10-07；未把未来升学计划计为已实现成果。

各目录说明个人分工、原始来源、运行前提与验证边界。完整文件路径和个人证明保存在本地。公开仓库不含真实密钥、学籍证件、生产数据、模型权重或第三方论文全文；团队和上游内容保留原归属。详见[归档说明与验证记录](docs/publication-notes.md)。
