# 麦积山数字人文平台 / Maijishan Platform

计算机专业团队课程项目的精选工程快照，包含 FastAPI 后端、React 前端、MySQL 数据访问、文献管理、检索与 AI 接口。

李卿阳在团队反思材料中记载的工作包括组长协调、文献爬取与整理、数据库维护及后续网站搭建。平台由团队协作完成，不代表所有模块均为个人独立原创。现存本地 Git 提交覆盖 2026 年 2—3 月；此公开快照整理于 2026-10-07。

## 内容

- `app/`：API、身份认证、文献与社区服务、检索模块。
- `frontend/`：React/Vite 界面源码和依赖锁文件。
- `tests/`：原工程测试。

## 运行前提

Python 3.11+、Node.js、MySQL；配置项见 `app/core/config.py`。配置读取环境变量。需要自行准备数据库结构、上传存储及可选 AI 服务配置。本快照不含生产数据库、文献全文、用户数据、真实密钥或部署服务器设置，不能直接恢复线上环境。

```sh
python -m pip install -e .
# 配置 DB_HOST、DB_PORT、DB_USER、DB_PASSWORD、DB_NAME、DB_CHARSET、
# UPLOAD_DIR、JWT_SECRET 及 config.py 中的其他必填环境变量后：
cd frontend
npm ci
npm run build
cd ..
# 创建 static/shouye 空目录（PowerShell: New-Item -ItemType Directory -Force static/shouye）
mkdir -p static/shouye
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

本次归档进行了源码语法与敏感信息检查，未重建生产数据库或重新验证线上运行。

公开快照的 Dockerfile 已去除生产数据拷贝并改为构建前端；尚未在本次归档中执行镜像构建。旧首页所需图片与资产未包含，相关旧页面会缺图；源项目保留完整本地内容。
