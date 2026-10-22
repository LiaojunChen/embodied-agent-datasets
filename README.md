# 具身 Agent 训练数据集调研

数据集清单、Hugging Face 下载量 / like 数与分阶段配比建议的静态页面。

- `data.py`：数据集清单与配比（改这里）
- `build.py`：拉取 HF 统计并生成 `site/index.html`（`--offline` 使用缓存的 `stats.json`）
- `sample.py`：为每个数据集抓一条真实样例写入 `samples.json`（图像缩略图存 `site/samples/`），本地运行后提交；CI 不运行
- `illustrative.py`：gated / 仅论文条目的格式示意
- `template.html`：页面模板

GitHub Actions 在每次 push 和每周一自动重新拉取统计并部署到 GitHub Pages。
