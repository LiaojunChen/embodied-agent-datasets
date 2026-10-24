# 具身 Agent 训练数据集调研

首页（`site/index.html`）：按七个能力方向（Grounding / Understanding / Identification / WorldModeling / Decision / Agentic / General）分类的数据集、token 统计、按 token 预算配比的数据合集版本（`site/mixes/*.json` 为各版本清单）。`site/v1.html` 是上一版清单。

- `catalog.py`：方向定义与数据集目录（每个条目一个可清洗单元：方向、HF 仓库、计数文件、视觉 token 预算）
- `tokens.py`：本地运行，抽样估算每个条目的文本 / 视觉 token → `cache/tokens.json`；`export_tokens.py` 精简为提交的 `tokens.json`
- `mix.py`：按方向预算生成配比版本（单一大源优先、单个数据集占比上限、不足如实标缺口）
- `sample2.py`：每个条目抓一条真实样例 → `samples_v2.json`（缩略图存 `site/samples/`）
- `build2.py`：拉取 HF 下载量 / like，渲染首页与清单；`build.py` 渲染上一版清单（`data.py` / `sample.py` / `samples.json`）
- `overrides.json`：gated 数据集的 token 估算依据；`illustrative.py`：gated / 仅论文条目的格式示意

GitHub Actions 在每次 push 和每周一刷新 HF 统计并部署。token 与样例在本地重新统计后提交：`tokens.py` → `export_tokens.py` → `build2.py`。
