# Crypto 5m 因子循环 RUNBOOK

配套 `DESIGN.md`（契约）。所有命令默认在 fork 根目录执行；Mac 用 `.venv/bin/python`，3060 用 `~/miniconda3/envs/rdagent-crypto/bin/python`。
LLM 循环只在 3060 上跑，Mac 只做建数据、跑测试、同步。

## 0. 一句话流程

```
建面板(Mac) -> 跑测试(Mac) -> sync_from_mac.sh 同步到 3060 -> setup_3060.sh -> 填 deploy/crypto/.env -> 冒烟 -> run_loop.sh 1 -> 看台账
```

## 1. 建面板（Mac / 3060 都一样）

```bash
# Mac：源数据是 1m csv.gz，输出到 git 仓库之外
.venv/bin/python -m rdagent.scenarios.crypto.data.build_panel \
    --source-dir /path/to/market-data/quant_assist/data/binance_spot_1m_core43_180d/normalized \
    --out-dir /path/to/rdagent-crypto-data/5m/full \
    --debug-out-dir /path/to/rdagent-crypto-data/5m/debug
# 可选：--bar-minutes 5 --debug-symbols 10 --debug-days 14
```

产物：`full/crypto_5m.h5 + README.md`（43 币全量）和 `debug/crypto_5m.h5 + README.md`（成交额前 10 币、最后 14 天）。
命令会打印行数和日期跨度，记下来（后面比对用）。3060 上路径换成 `~/rdagent-crypto-data/5m/{full,debug}`。
一般不在 3060 上重建，直接用 `sync_from_mac.sh` 把面板同步过去。

## 2. 跑测试

```bash
.venv/bin/python -m pytest test/scenarios/crypto -q            # 全部（Mac）
.venv/bin/python -m pytest test/scenarios/crypto/test_runner_feedback.py -q
```

测试不需要 API key、Docker、真实面板（用临时合成面板）。3060 上把 python 换成 conda env 的即可。

## 3. 同步到 3060

```bash
bash deploy/crypto/sync_from_mac.sh <ssh-host>     # 例如 user@192.168.1.50 或 ~/.ssh/config 里的别名
```

- fork → `~/rdagent-crypto`（排除 `.git .venv git_ignore_folder log pickle_cache __pycache__ *.egg-info`，也不带 `deploy/crypto/.env`）。
- `/path/to/rdagent-crypto-data/5m` → `~/rdagent-crypto-data/5m`。脚本会打印两边大小。
- 改了代码再跑一次即可（tar 覆盖，幂等）。

## 4. 3060 环境（一次性，可重复执行）

```bash
ssh <host>
cd ~/rdagent-crypto && bash deploy/crypto/setup_3060.sh
```

用户级安装：miniconda → `~/miniconda3`，conda env `rdagent-crypto`（py3.11），`pip install -e .` + pandas/numpy/tables/pyarrow/scipy/pytest，最后 import 自检。不需要 sudo，不需要 Docker。

## 5. 填 `.env`

```bash
cp deploy/crypto/env.example deploy/crypto/.env && vi deploy/crypto/.env
```

必须改的：
- 推荐 `ANTHROPIC_API_KEY` + `CHAT_MODEL=anthropic/claude-sonnet-5`（litellm 对 `anthropic/` 前缀支持结构化输出；`openai/` 前缀下 Claude 的 response_format 会被丢弃，只能靠 prompt + 重试）。走 new-api 网关时可加 `ANTHROPIC_API_BASE=http://127.0.0.1:3000`。备选 `CHAT_MODEL=openai/...` + `OPENAI_API_BASE` + `OPENAI_API_KEY`。`REASONING_EFFORT=` 留空。
- `FACTOR_CoSTEER_python_bin` = conda env 的 python 绝对路径（`setup_3060.sh` 最后会打印）。

默认即可的：
- `CRYPTO_LOCAL_EMBEDDING=1`：不用 embedding 服务，CoSTEER 的知识检索退化为词袋相似度（只影响“找相似历史任务/报错”的质量，不影响主 LLM）。有 embedding 模型时改成 `EMBEDDING_MODEL=...`。
- `FACTOR_CoSTEER_data_folder{,_debug}`、`PICKLE_CACHE_FOLDER_PATH_STR`、`LOG_TRACE_PATH`、`CRYPTO_EVAL_ledger_path` 都指向 `~/rdagent-crypto-data/...`。
- `CRYPTO_EVAL_*` 是评估分段/闸门阈值，`CRYPTO_LOOP_*` 是组件类路径，一般不动。

## 6. 冒烟（按顺序，每步几秒）

```bash
cd ~/rdagent-crypto
set -a; . deploy/crypto/.env; set +a
PY=$FACTOR_CoSTEER_python_bin

# 1) 数据文件在不在
ls -la $FACTOR_CoSTEER_data_folder $FACTOR_CoSTEER_data_folder_debug

# 2) 场景能不能实例化（会校验两份面板存在，并渲染所有 prompt；不调 LLM）
$PY -c "from rdagent.scenarios.crypto.experiment import CryptoFactorScenario as S; s=S(); print(s.get_scenario_all_desc()[:800])"

# 3) 循环组件类都能 import
$PY -c "from rdagent.app.crypto_rd_loop.conf import CRYPTO_PROP_SETTING as P; from rdagent.core.utils import import_class; [import_class(getattr(P,k)) for k in ('scen','hypothesis_gen','hypothesis2experiment','coder','runner','summarizer')]; print('ok')"

# 4) CLI 命令注册了
$(dirname $PY)/rdagent crypto_factor --help

# 5) 测试
$PY -m pytest test/scenarios/crypto -q

# 6) 离线端到端冒烟（不调 LLM）：两个手写因子跑完整链路
#    因子子进程执行 -> RankIC 评估/门槛 -> 反馈(假 LLM) -> 台账 report -> 循环组件类解析 -> 假设 prompt 渲染
#    读 $HOME/rdagent-crypto-data/5m/{full,debug}，可用 --full-folder/--debug-folder 覆盖；末行应打印 SMOKE OK
$PY deploy/crypto/smoke_offline.py
```

注意：`python -m rdagent.app.crypto_rd_loop.factor --help` 会被 fire 当成 `help=True` 参数直接进入 `main()`（上游
`qlib_rd_loop.factor` 同样如此）；要看 fire 帮助请用 `python -m rdagent.app.crypto_rd_loop.factor -- --help`，
正式入口是 `rdagent crypto_factor --help`。

`rdagent health_check` 会去检查 Docker，在 3060 上大概率报 Docker 相关错误——**忽略**，本场景不需要 Docker（因子代码由 `FACTOR_CoSTEER_python_bin` 子进程执行，评估是纯 pandas）。需要的话 `rdagent health_check --no-check-docker` 只查端口/环境。

## 7. 跑 1 轮

```bash
bash deploy/crypto/run_loop.sh 1          # 参数 = loop_n（会话的绝对轮数，续跑时包含已完成的轮），默认 5
```

一轮 = propose → exp_gen → coding(CoSTEER 在 debug 面板上演化代码) → running(全量面板上 RankIC 评估) → feedback。
日志和会话在 `$LOG_TRACE_PATH/`，会话快照在 `$LOG_TRACE_PATH/__session__/<loop>/<step>_<name>`。
每个实验的工作区（`$WORKSPACE_PATH/<uuid>/`）里有 `crypto_eval.json`（完整评估）和 `factor_metrics.csv`（因子 × 分段 × 期限 一行）。

## 8. 看台账

```bash
$PY -m rdagent.scenarios.crypto.ledger report --path $CRYPTO_EVAL_ledger_path
```

输出：已入库因子表（valid 15m 的 ic_mean / ic_ir / ic_tstat、rank_autocorr_1、test 15m ic_mean）+ 按 valid ic_tstat 排前 10 的被拒因子及原因（`R0..R4`；`R0` = 前视/退化：截断面板重跑后数值变化、valid |ic_mean| 超过 `CRYPTO_EVAL_gate_max_abs_ic`、或 IC 序列常数；`ic_tstat` 为 Newey-West 统计量，另有 `ic_tstat_naive` 与去掉每个币种 train 均值后的 `ic_tstat_dynamic`）。
台账是 JSONL，每评估一次追加一行，`jq` 直接读也行：`tail -1 $CRYPTO_EVAL_ledger_path | jq '.accepted, .library_after'`。

## 9. 续跑会话

```bash
ls $LOG_TRACE_PATH/__session__/            # 找最新 loop 号
# loop_n 是绝对轮数：已完成 2 轮后想再跑 3 轮要传 5；传 <= 已完成轮数会立刻 "Reach stop criterion" 退出
bash deploy/crypto/run_loop.sh 5 --path "$LOG_TRACE_PATH/__session__/<loop>/<step>_<name>"
# 或只走一步：... --path <session> --step-n 1
```

`--checkout/--no-checkout`（默认 checkout）控制是否从该快照重放；`LOG_TRACE_PATH` 固定时新启动会复用同一个 trace 目录，不固定则每次新建。

## 10. 面板变了必须做的事

`FactorFBWorkspace.execute` 的 pickle 缓存 key 只含 `data_type + factor.py 源码`，**不含数据路径/内容**。所以只要重建了 `crypto_5m.h5`（换日期、加币、改聚合）：

1. 把 `.env` 里的 `PICKLE_CACHE_FOLDER_PATH_STR` 改成新目录（如 `..._v2`），否则旧因子值会被复用；
2. 如需要，把 `CRYPTO_EVAL_train/valid/test_*` 改成新跨度；
3. 从新会话开始（不要 `--path` 续旧会话，旧会话里的库因子结果对应旧面板）。

## 11. 常见问题

- `FileNotFoundError: ... crypto_5m.h5`：`FACTOR_CoSTEER_data_folder(_debug)` 路径错或没同步；场景初始化时会直接报，不会去生成 Qlib 数据。
- `FactorEmptyError`：本轮所有因子在全量面板上执行失败/无有效列，循环会跳到 feedback 继续，不是崩溃。看工作区里的 `factor.py` 和 stderr。
- LiteLLM 报 `reasoning_effort` 不支持：`.env` 里 `REASONING_EFFORT=` 必须为空。
- embedding 报错：确认 `CRYPTO_LOCAL_EMBEDDING=1` 已生效（日志里有 `local_embedding: ... installed`）。
