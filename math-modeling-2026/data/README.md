# 数据来源与文件用途

建模输入为赛题官方附件 1、附件 2，物性经验公式、初始条件和几何参数来自 A 题题设及附录。

## 官方输入数据

| 文件 | 内容 | 数据范围与单位 |
| --- | --- | --- |
| `附件1.xlsx` | 烘房温度与水分浓度，共 241 条记录 | 时间 0—14400 s；温度 °C；水分浓度 kg/kg |
| `附件2.xlsx` | 药材外半径，共 145 条记录 | 时间 0—259200 s；半径 cm |

运行时将两个官方工作簿放在同一个目录中，并从支撑材料根目录执行：

```sh
python -m pip install -r requirements.txt
python code/run_all.py --data-dir "官方附件所在目录"
```

完整数值检查与配图流程为：

```sh
python code/run_all.py --data-dir "官方附件所在目录" --checks --plots
```

程序先校验官方文件的 SHA-256、数据维度和时间顺序，再生成 `processed/attachment1.npy` 与 `processed/attachment2.npy`。文件校验结果写入 `results/input_validation.json`，计算结果写入 `results/` 和 `results/raw/`，配图输出到 `figures/recreated/`。

## 本目录文件

| 文件或目录 | 内容与用途 |
| --- | --- |
| `paper_tables.json` | 结果核验使用的表格数据，由 `code/verify_results.py` 读取 |
| `figure_tables.json` | 可视化使用的表格数据快照，由 `code/plot_paper_figures.py` 读取 |
| `processed/` | 根据官方工作簿自动生成的数值缓存 |

表格数据用于结果对照和可视化；温度场、水分场及半径演化由求解程序依据官方输入与题设参数计算。本精选快照不包含原结果工作簿或大体积 CSV；需要时运行求解与导出程序重新生成。
