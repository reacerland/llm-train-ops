# llm-train-ops

LLM 训练 CI/CD 工具集。当前包含 `fetch` 数据获取工具。

## 安装

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## fetch：从 ModelScope 拉取数据集到本地缓存

```bash
trainops fetch --dataset AI-ModelScope/xxx \
    --subset default --split train \
    --max-size 500MB \
    --cache-dir ~/.cache/trainops
```

- 流式拉取，按字节配额（`B/KB/MB/GB`，1024 进制）停止，不全量下载
- 原样保存：每条记录完整 JSON 序列化为 `data.jsonl` 一行
- 缓存增量复用：同一数据集再次请求更大配额时只拉差额；中断后重跑自动续拉
- 缓存位置：`<cache-dir>/datasets/<dataset-slug>/<subset>/<split>/`（`data.jsonl` + `meta.json`）
- `--subset` 可选：不带 named config 的数据集省略该参数即可（缓存归到 `default/` 目录）

## 测试

```bash
pytest             # 单元测试（默认跳过 slow）
pytest --runslow   # 含真实 ModelScope 拉取的集成测试
```
