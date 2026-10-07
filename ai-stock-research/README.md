# AI 股票信息研究系统 / AI Stock Research System

个人工程探索的精选源码快照。原本地 Git 仓库在 2026-05-31 至 2026-06-02 留有 Li Qingyang 提交，含命令行、数据来源适配、证据追踪、研究卡片、规则评分与测试。此仓库不声称投资收益、真实交易业绩或金融机构实习经历。

公开内容仅为 `src/ai_stock_discovery`、`tests` 与包配置，不含真实 API key、持仓、交易记录、本地数据库、采集结果或研究对象清单。代码中的 `secret` / `secret-key` 为测试桩占位值，生产来源使用环境变量。

```sh
python -m pip install -e .
python -m ai_stock_discovery.cli --help
python -m unittest discover -s tests
```

联网来源需要自行配置身份和服务环境变量；部分测试依赖未公开的数据模板。此归档不构成投资建议，未在本次盘点中运行联网采集或交易。
