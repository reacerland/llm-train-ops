# llm-train-ops

LLM 训练 CI/CD 工具集。当前包含 `fetch` 数据获取工具、`convert` 缓存转 HF Arrow 工具与 `mix` 多源数据混合打包工具。

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

## mix：多源数据混合打包成训练集

```bash
trainops mix --config mix.toml --cache-dir ~/.cache/trainops
```

- 配置文件为 TOML：声明多个数据源（各指向 `convert` 产物的缓存别名）与混合比例、tokenizer、`max_seq_len`、`seed`
- 文档级加权不放回抽样：按 `ratio` 从各源抽文档，编码后补 EOS 拼接成连续 token 流，按 `max_seq_len` 硬切成定长序列；某源抽完即停止并报告（产物可用，exit 0）
- tokenizer 两种形态：预置名 `"llama2"`（需 `uv pip install -e ".[hf]"`，vocab 32000），或自定义 Python 类文件（鸭子类型接口：`eos_token_id` / `vocab_size` / `encode(text) -> list[int]`，不添加特殊符号）
- 固定 `seed`（默认 42）保证同配置重跑产物逐字节一致
- 产物为训练侧零加工格式：`tokens.bin`（uint16/uint32 扁平定长序列）+ `meta.json`（各源配置 vs 实际占比、耗尽源等）

配置示例：

```toml
tokenizer = "llama2"
max_seq_len = 4096
seed = 42

[[source]]
name = "web"
dataset = "my-web"   # convert 时的 --name 缓存别名
ratio = 0.7

[[source]]
name = "code"
dataset = "my-code"
ratio = 0.3
```

训练侧用法：

```python
import json, numpy as np
meta = json.load(open("<output>/meta.json"))
tokens = np.memmap("<output>/tokens.bin", dtype=meta["dtype"], mode="r")
seqs = tokens.reshape(-1, meta["max_seq_len"])  # 一条序列: input_ids=seq[:-1], labels=seq[1:]
```

## 测试

```bash
pytest             # 单元测试（默认跳过 slow）
pytest --runslow   # 含真实 ModelScope 拉取的集成测试
```
