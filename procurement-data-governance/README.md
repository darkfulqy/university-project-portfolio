# 高校采购数据治理与公开公告采集

2026 年高校采购项目及“智采通”校内创新大赛经历的工程补充。本人担任项目组长，参与架构、任务拆分、数据治理与答辩；这些角色来自成长报告。这里保留的是本地分工的公告采集代码和知识图谱规范，不代表整个团队平台。

## 保留内容

- 多地区采购公告的分页采集、字段提取、附件分类、断点与去重处理。
- `bulk_tenders.py`、`collect_announcements.py` 和共享下载模块；省份专用入口只是短包装，本快照使用统一入口。
- 三份原有测试和[知识图谱格式规范](docs/knowledge-graph-schema.md)。

## 使用与状态

原工程使用 Python 3.10+，依赖见 `requirements.txt`。可先安装依赖，再阅读 `download_site.py --help` 或 `collect_announcements.py --help`。

```sh
python -m pip install -r requirements.txt
python -m unittest test_announcements test_bulk_tenders test_network_route -v
```

以上为原工程的验证入口，本次归档只做静态审查，没有运行采集或重跑测试。`network_route.json` 已替换为默认关闭的通用配置。正式采集前应检查目标站点、请求间隔和磁盘参数。

Excel 导出使用 Node.js 与 `@oai/artifact-tool`，该运行时未随包分发；没有相应环境时使用 `--no-excel` 保存通用 JSON。原始标书库、网站响应、数据库、会议纪要、成员资料和竞赛报名文件仅在本地留档。本次不复核源站当前可用性，不把“有采集代码”写成“所有数据已采全”。

来源：本人采购项目的 2026 年 9 月本地留档，以及 2026 年 10 月成长报告。私有路径与哈希映射单独保存在本机。
