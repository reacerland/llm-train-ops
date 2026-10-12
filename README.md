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
    --name my-dataset \
    --subset default --split train \
    --max-size 500MB \
    --cache-dir ~/.cache/trainops
```

- 流式拉取，按字节配额（`B/KB/MB/GB`，1024 进制）停止，不全量下载
- 原样保存：每条记录完整 JSON 序列化为 `data.jsonl` 一行
- 缓存增量复用：同一数据集再次请求更大配额时只拉差额；中断后重跑自动续拉
- `--name` 可选缓存别名：指定后缓存键用别名（`<cache-dir>/datasets/<name>/...`），方便流水线按名字引用；缺省用数据集 ID 的 slug
- `--subset` 可选：不带 named config 的数据集省略该参数即可（缓存归到 `default/` 目录）
- 拉取过程显示进度：TTY 下为 tqdm 进度条（百分比/速率/条数），非 TTY（CI）下按 checkpoint 打印文本进度

## 测试

```bash
pytest             # 单元测试（默认跳过 slow）
pytest --runslow   # 含真实 ModelScope 拉取的集成测试
```
