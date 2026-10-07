# Legal RAG Competition Prototype

新致杯 AI 创新大赛相关团队历史原型的精选源码，用于审阅当时的文书读取、结构化抽取、向量检索和生成流程。文件留痕集中于 2025 年 6—7 月；整理日期为 2026-10-07。这是留存版本，不代表当年最终交付或当前可直接运行的完整系统。

## Attribution and evidence

李卿阳的本地个人材料将此项目列为竞赛经历，自述参与资料整理、知识库构建和流程调试。不同简历版本对成员/组长的表述不一致；具体个人分工、进入决赛及优胜奖仍需独立团队记录、名单或证书核实。源码存在可以证明留有工程，不单独证明获奖。这里保留团队项目属性，不把全部源码宣称为李卿阳个人独立原创。

## What is included

- `app.py`：三个 DOCX 文件读取与合并的 Flask 界面原型。`simulate_external_program` 是原有模拟占位函数，不执行 RAG 或大模型生成。
- `final_all(6).py`：另一份较完整的集成尝试；保留历史文件名，包含 Qwen 文本结构化、Ollama 向量、Chroma 检索和生成流程。它与 `app.py` 是两个独立入口，不会自动相互调用。
- `templates/index.html`：原 HTML 模板，只有空白表单和占位文案，没有真实案件或个人信息。
- `requirements.txt`：本次根据 import 推导的未锁定依赖列表，不是原环境锁文件。
- `PUBLICATION_CHANGES.json`：原文件和公开文件的哈希，以及每一处脱敏改动。不记录原凭据值。

数据库、案件全文、案例附件、客户数据、运行日志、背景音乐、截图和模型文件均未包含。原代码中的两个 API 凭据位置已改为 `os.environ["DASHSCOPE_API_KEY"]`，没有后备硬编码密钥。

## Environment

界面占位入口依赖 Flask 和 python-docx。集成入口还依赖 requests、chromadb、openai、Windows pywin32，以及用于读取 `.doc` 的 Microsoft Word。它通过回环地址 `127.0.0.1:11434` 调用本地 Ollama 的 `nomic-embed-text:latest`；大模型 endpoint 为代码中原有 DashScope 兼容接口。

调用集成入口之前，需要在进程环境中设置 `DASHSCOPE_API_KEY`。代码不会自行读取 `.env`，未设置变量会报错。还需要自行准备 `db/chroma_db` 下的 `test_1` 集合、匹配的向量维度、所需 metadata 字段及可访问的文档路径。此公开快照没有建库脚本和语料，不能据此直接复原原知识库。

## Known limitations

- 本次仅通过 Python AST 语法解析、HTMLParser 读取和模板分隔符配对检查，未安装项目依赖、完整渲染模板、启动应用、调用 Word/模型 API 或运行检索。
- `final_all(6).py` 已调用本文件中的 `search`；未纳入旧版本里名称不同的 `search_chromadb` 调用。本版本仍需外部数据库和服务才能继续处理。
- 结构化提示词要求“指控罪名”，检索读取“控诉罪名”，原字段不一致仍然保留。
- 路由读取 `reference_texts`，但现存处理函数返回的是 `message` 和 `img_srcs`，参考文本框可能保持占位内容。全局参考路径列表也没有按请求清空。
- 图片与背景音乐未随源码公开，相关资源会缺失；Google Fonts 字体依赖外部访问。
- 原 Flask 入口保留开发调试设置、直接读取服务器文件路径及打印处理内容的行为。它只适合审阅和本地合成资料实验，不是供公网部署的服务；不要给它输入客户或真实案件敏感资料。

历史功能缺口没有在归档中补写。只有凭据读取发生了脱敏修改，具体位置见变更清单。未添加覆盖团队贡献或第三方内容的开源许可。
