# RD-Agent 的加密行情因子研究扩展

个人探索：在 [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) 上增加加密现货分钟级因子的研究场景、评估和循环入口。上游框架属于 Microsoft 及其贡献者；本目录只收录本机新增的 crypto 扩展文件，并保留 [上游 MIT 许可](LICENSE.upstream)。

## 本地新增内容

- `rdagent/scenarios/crypto/`：数据面板、场景配置、提示模板、评价、反馈与台账。
- `rdagent/app/crypto_rd_loop/`：复用上游循环的入口。
- `test/scenarios/crypto/`：数据构建、评价、提示、台账和循环相关的原有测试。
- `docs/crypto_5m/`：原阶段设计契约与运行手册，本机路径已替换为通用示例。

设计记录使用 5 分钟观测起点与 5／15 分钟远期收益，通过横截面 Spearman RankIC 等指标研究因子评价。代码依赖上游的实验、日志、CoSTEER 与循环机制，不能把这部分包装为从零实现的完整平台，也不等同于可盈利的交易系统。

## 依赖与状态

原设计记录基于 RD-Agent `32b3d395`，Python 3.11。复核时应先自行取得该上游版本并按上游说明安装，再将本目录三个新增代码／测试子目录合入相同位置；保留上游其余文件。入口可用：

```sh
python -m rdagent.app.crypto_rd_loop.factor --help
python -m pytest test/scenarios/crypto -q
```

本次没有复制上游 CLI 的修改，而是使用新增模块入口。原运行手册涉及的部署脚本、真实面板、环境变量与运行日志未公开，手册因此仅作为历史设计证据，不能视作本精选目录独立部署承诺。实际运行因子循环需要另行配置数据及模型服务，可能产生 API 费用。

本次只检查文件来源和静态内容，未运行研究循环或重做回测，也未验证原设计中的收益／优胜标准已达到。来源为成长报告、本地扩展及其设计说明。
