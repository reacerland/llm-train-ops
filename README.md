# llm-train-ops

LLM 训练 CI/CD 工具集。包含 `fetch` 数据获取、`convert` 缓存转 HF Arrow、`mix` 多源数据混合打包与 `train` 模型训练命令。

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
- 预置 tokenizer 只从本地 HuggingFace 缓存加载（`local_files_only=True`），不会联网；缓存中不存在时报错并提示先手动下载
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

## train：任务式模型训练

一个任务配置 `task.toml` 包含训练任务的全部配置：混合数据集定义（`[data]`，同 mix 配置）、模型 Python class 文件（`[model]`）、训练超参（`[train]`）与可选的推理 prompt（`[prompts]`）。

CI 流水线三步：

```bash
# 1. 创建训练任务 = 编写 task.toml（见下）
# 2. 创建混合数据集(mix 直接读 task.toml 的 [data] 段)
trainops mix --config task.toml
# 3. 训练,产出保存到任务产出目录
trainops train --config task.toml --output runs/task-001
```

任务配置示例：

```toml
[data]
tokenizer = "llama2"          # 或 { class = "path/to.py:ClassName" }
max_seq_len = 4096
output = "data/mixture"       # 可选;缺省 <cache>/mixtures/<配置名>

[[data.source]]
name = "web"
dataset = "my-web"
ratio = 0.7

[[data.source]]
name = "heldout"
dataset = "my-valid"
split = "validation"
ratio = 1.0
role = "valid"                # 只做验证，绝不进训练数据

[model]
file = "models/gpt.py"        # 相对配置文件目录解析
class = "GPT"
[model.args]                  # 原样传给模型构造函数
vocab_size = 32000
d_model = 768
n_layer = 12

[train]
total_tokens = 100_000_000    # 训练多少 token
batch_size = 8
lr = 3.0e-4
warmup_steps = 200
log_interval = 10
save_interval = 500
eval_interval = 500           # 可选:每 N 步评估一次 PPL/top-k;缺省不评估
eval_batches = 20             # 每次评估取多少 batch(固定 seed 选取,可复现)
seed = 42

[prompts]                     # 可选:训完用最终权重生成样本
texts = ["今天天气"]
max_new_tokens = 50
```

- 模型为鸭子类型接口：普通 `torch.nn.Module`,`forward(input_ids) -> logits [B, T, V]`,loss 由训练脚本计算(input_ids=seq[:-1], labels=seq[1:])
- 数据源为 mix 产物(`tokens.bin` + `meta.json`),memmap 加载,同 seed 重跑数据顺序一致;`total_tokens` 超过数据集总量时自动多 epoch
- CUDA 上自动 bf16 autocast,否则 fp32;单卡单进程
- 产出目录内容:`model_final.safetensors`(最终权重)、`loss.jsonl`(每 log_interval 一步的 step/loss/lr/吞吐)、`checkpoint.pt`(每 save_interval 覆盖更新,含 optimizer 与 RNG 状态)、`samples.txt`(配了 [prompts] 才有)、`task.toml`(配置快照)
- 断点续训:`trainops train --config task.toml --output runs/task-001 --resume` 从产出目录的 checkpoint.pt 恢复,轨迹与一次性跑完一致;`--resume` 与 `--overwrite` 互斥
- 冒烟与正式训练同一条命令,只是两份配置;冒烟看 loss.jsonl 与 samples.txt 人工确认后再跑正式

## plot：训练曲线可视化

```bash
trainops plot --input runs/task-001/eval.jsonl \
    --x step --y loss,ppl,top1,top5 \
    --output curves.png     # 可选;缺省 <eval.jsonl 同级>/eval_curves.png
```

- 读取 train 产出的 `eval.jsonl`,train/valid 两条曲线画在同一坐标系,直接对照是否过拟合
- `--y` 逗号分隔多个指标时生成纵向堆叠的子图(共享 x 轴);`--x`/`--y` 可指定 eval.jsonl 中的任意数值字段(如 `step`/`loss`/`ppl`/`top1`/`top5`)
- 字段不存在时报错并列出可用字段;无头 CI 环境直接出 PNG,无需显示服务

## 测试

```bash
pytest             # 单元测试（默认跳过 slow）
pytest --runslow   # 含真实 ModelScope 拉取的集成测试
```
