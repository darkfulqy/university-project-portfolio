# EEG 数据蒸馏与合成训练效用

这是从个人研究工作区整理的精选源码快照，保留 AI 辅助开发痕迹与上游方法归属。它不是完整实验归档，也不是已发表成果。整理时未执行训练、仿真或重新计算科研结果；以下数字均为原工作区历史记录。

这个方向研究少量合成或筛选 EEG 是否保留训练价值，并用统一骨干、预算、被试划分和真实子集对照核验结论。个人工作主要为复现驱动、接口适配、对照设计和结果审查；EEG-DLite、LGM、TGA、UniTSC、JET 均是上游方法。

- [EEG-DLite 复现驱动](eeg-dlite/)：准备、筛选、训练、汇总与共享模块。
- [LGM→EEG 试验选段](evaluation/repro/lgm/)：迁移和复核代码，需完整原工作区依赖。
- [JET 指标数值核验](research/2026-09-30/jet_metric_sanity.py)。
- [历史结果与限制](HISTORICAL_FINDINGS.md)。
- [上游固定版本](UPSTREAM_VERSIONS.txt)与[依赖记录](requirements-mac.txt)。

## 运行范围

本快照不含 `third_party/EEG-DLite`、LGM 官方库、教师权重、预处理数据或实验产物。`eeg-dlite/common.py` 明确从相邻 `third_party/EEG-DLite` 导入官方方法；其六个驱动文件虽成组保留，仍需原库和数据才能运行。LGM 与 JET 文件是精选模块，应回到原工作区按已冻结的协议运行，本页不提供未经验证的一键命令。

EEG-DLite 的本地试验使用 BCI IV-2a 替代需另行授权的 SEED，并采用小网络 pilot 协议；不能称为原论文全部实验的等价复现。NN 结构和筛选算法归原作者，代码中的来源说明保留。上游许可证仍由原仓库管理，本包没有重新授权上游代码或数据。
