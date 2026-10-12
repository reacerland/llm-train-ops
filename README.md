# llm-train-ops

LLM 训练 CI/CD 工具集。当前包含 `fetch` 数据获取工具与 `convert` 缓存转 HF Arrow 工具。

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

## convert：缓存数据集转 HF Arrow 格式

```bash
trainops convert --dataset AI-ModelScope/xxx \
    --name my-dataset \
    --subset default --split train \
    --fields text,meta \
    --cache-dir ~/.cache/trainops
```

- 定位与 `fetch` 完全相同的缓存条目（`--dataset`/`--name`/`--subset`/`--split`），把 `data.jsonl` 转换为 HF datasets 目录
- 默认输出到缓存并排目录 `<cache-dir>/datasets/<name>/<subset>/<split>/data.arrow`，可用 `--output` 覆盖；已存在时报错，`--overwrite` 清空重建
- `--fields` 必填：逗号分隔的字段白名单。字符串原样保留，数字/布尔/数组/对象序列化为 JSON 字符串，所有列为 Arrow `string` 类型，便于直接接 tokenizer
- 缺字段或非法 JSON 行默认报错退出；`--skip-bad` 跳过并统计
- 分批流式转换（每批 1000 条写入 Arrow RecordBatch），GB 级缓存内存占用恒定
- 训练侧用法：`datasets.load_from_disk("<...>/data.arrow")` 后直接 `.map()` tokenize

## 测试

```bash
pytest             # 单元测试（默认跳过 slow）
pytest --runslow   # 含真实 ModelScope 拉取的集成测试
```
