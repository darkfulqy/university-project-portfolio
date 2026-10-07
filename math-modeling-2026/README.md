# 2026 年数模国赛 A 题：柱状中药材烘干

这是 2026 年 9 月数模国赛团队作品的代码精选。成长报告记录本人参与水热耦合建模、有限差分求解、收缩模型比较及数值检查；本地材料不单独划分每个模块的个人作者，因此作为团队参赛成果展示。

## 模型与工程

- `code/common/solver.py`：一维径向热传导、水分扩散和收缩模型。
- `code/Q1/`—`code/Q4/`：四问入口；第四问比较质量坐标与随体坐标。
- `code/Q1/end_effects.py`、`code/checks.py`：二维端部效应、参数敏感性与网格检查。
- `code/export_results.py`、`code/verify_results.py`：结果导出和参考表格逐项对照。
- `results/`：原终稿支撑包已有的数值摘要与一致性记录，不是本次重新计算的结果。

历史摘要中的第四问烘干时长为 **51.114951 小时**。同时，原始一致性记录明确显示 **225 项对照中 126 项匹配，`all_values_match=false`**。因此本快照不宣称所有论文表格都已完成一致性验证；差异逐项保存在 `results/consistency_check.csv`。

## 复现入口与边界

原环境为 Python 3.12。自行取得赛题官方附件 1、附件 2，并将两份工作簿放在同一目录后：

```sh
python -m pip install -r requirements.txt
python code/run_all.py --data-dir /path/to/official-attachments
# 可选：追加 --checks --plots，进行附加数值检查和配图
```

输入读取器会验证附件的校验值和形状。官方附件、下载论文、第三方 LaTeX 模板、重复终稿、压缩包及大型数组未包含；计算过程中会重新生成 `data/processed/` 和结果数组。`data/figure_tables.json` 与 `paper_tables.json` 是原工程用于配图、比对的表格快照。

本次仅静态检查源文件与历史输出，未执行求解器，也没有替原始成果修改结论。来源为 2026-09-13 “论文最新合并版”支撑材料与成长报告；原始路径及文件哈希保留在私有索引。
