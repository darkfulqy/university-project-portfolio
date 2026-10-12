# AI Agent 与 RAG 实习经历

2026 年 7 月至 8 月 22 日在杭州长亮科技担任实习工程师（AI Agent 开发），参与 AI Coding 平台后端：

- Agent SDK 的会话管理、事件流与异常处理；对照 OpenCode TypeScript 源码与 OpenAPI Schema 升级 Python SDK，补充旧版事件/消息/错误兼容层与前向兼容。
- 为公司开发可被大模型自主调用的文件 RAG 服务：FastAPI + Qdrant，多格式文档解析（PaddleOCR-VL）、500 token 滑动窗口切分与按源文件聚合召回；提供 MCP 接口、Agent Skill，以及经 OpenAI Responses API Function Calling 实现 Agentic RAG 的 CLI。
- 在 9 条公开财报召回用例上，扩大候选窗口后文件与证据 Recall@K 由 88.9% 提升至 100%。

相关设计文档（多格式解析、Qdrant 滑动窗口与源文件召回、RAG Demo）和工程目录留存于本地；Haystack 与 RAGFlow 仅作为参考源码阅读。

这是公开经历摘要。公司内部代码、设计原件、内部远程地址、工作数据和入职个人材料均未复制到本仓库。证据路径仅存于本机私有索引；如后续需要审核，应在适当授权下提供经过处理的材料。
