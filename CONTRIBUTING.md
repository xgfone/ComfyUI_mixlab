# 开发约定

本项目使用 [Ruff](https://docs.astral.sh/ruff/) 统一 Python 代码的检查、自动修复、
导入排序和格式化，并使用 [Oxfmt](https://oxc.rs/docs/guide/usage/formatter.html)
格式化 Markdown。编辑器与命令行会读取项目内的同一份配置。

## 本地安装

```shell
python -m pip install -r requirements-dev.txt
```

## 命令行

修复所有可安全修复的问题，然后格式化代码：

```shell
ruff check --fix .
ruff format .
```

只检查、不修改文件：

```shell
ruff check .
ruff format --check .
```

若本机命令行已提供 `oxfmt`：

```shell
oxfmt "**/*.md"
oxfmt --check "**/*.md"
```

## 编辑器

- VS Code：安装工作区推荐的 Ruff 与 Oxc 扩展。Python 文件由 Ruff 处理，Markdown 文件由 Oxfmt 处理。
- Zed：Ruff 支持已内置；另外从扩展面板安装 Oxc 扩展。项目设置会为 Python 和 Markdown 启用各自的保存时格式化。

自动的 `fix all` 不包含 Ruff 标记为不安全的修复；这类修改应单独检查后手动应用。
