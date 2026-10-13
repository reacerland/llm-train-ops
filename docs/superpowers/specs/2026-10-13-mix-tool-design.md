# trainops mix 多源数据混合打包工具 — 设计文档

日期：2026-10-13
状态：待批准

## 背景

`llm-train-ops` 已有 `fetch`（ModelScope → 本地 JSONL 缓存）与 `convert`（JSONL → HF Arrow）两步（见前两份设计文档）。本期新增 **`mix` 命令**：把多个 `convert` 产出的 Arrow 数据集，按一份 TOML 配置中声明的混合比例，用指定 tokenizer 编码、打包成定长 token 序列，落盘为训练循环可直接加载的产物。

设计目标：**训练侧零加工**。tokenize 与混合是"贵"操作，一次性做完；训练循环只需 memmap 加载，不需要 datasets 库、不需要在线 map。产物数据契约对自定义训练循环与任意训练框架（HF Trainer 等）一视同仁。训练循环本身本期不实现，后续另行设计。

## 关键决策（来自需求讨论）

| 决策点 | 结论 |
|---|---|
| 产物格式 | 扁平 token 二进制 `tokens.bin` + 元数据 `meta.json`，**不存文档边界**——训练时允许跨文档注意力 |
| 序列切分 | 连续 token 流按 `max_seq_len` 硬切；文档长于 `max_seq_len` 时自然横跨多条序列，不截断 |
| 尾部不完整序列 | 丢弃（计数并写入 meta），不足一条完整序列的余量不留 |
| dtype | `vocab_size < 65536` → `uint16`，否则 `uint32` |
| Tokenizer | 配置驱动，两种形态：预置名（`"llama2"`）或自定义 Python 类文件（`"path/to.py:ClassName"`） |
| 预置 tokenizer | `llama2` → `NousResearch/Llama-2-7b-hf`（非 gated 镜像，vocab 32000，uint16 最佳点），经 `AutoTokenizer` 加载 |
| 自定义 tokenizer 接口 | 鸭子类型：`eos_token_id: int`、`vocab_size: int`、`encode(text: str) -> list[int]`，不继承基类 |
| 混合粒度 | **文档级**加权抽样，**不放回**，权重 = 配置比例，全程固定不重归一化 |
| 抽干语义 | 任一源文档取尽 → 停止打包，明确打印耗尽源 + 实际/配置比例报告，**保留已产出，exit 0** |
| 确定性 | 固定 `seed`（默认 42），同一配置重跑产物逐字节一致 |
| EOS | 每篇文档编码后追加 `eos_token_id`，由流内 EOS 充当文档分隔 |
| 空文档 | tokenize 结果为空 → 跳过并计数，不插入 EOS |
| encode 语义 | 统一约定 `encode(text)` 不添加任何特殊符号（预置 tokenizer 内部用 `add_special_tokens=False`），工具只在文档末尾追加一个 EOS；BOS 不自动添加 |
| 配置文件 | TOML（`tomllib` 标准库解析，Python 3.10 用 `tomli` 兜底） |
| 实现结构 | 两阶段：先逐源预 tokenize 到临时 `.bin`（含文档偏移，仅中间态），再混合打包；临时文件运行结束清理；跨源增量复用留接口本期不做 |
| 单篇编码失败 | 报错退出（exit 1，指明源与记录序号）；本期不做 `--skip-bad` |

## 配置文件

```toml
tokenizer = "llama2"            # 预置名
# tokenizer = { class = "my_tokenizers/my_bpe.py:MyTokenizer" }   # 自定义类文件

max_seq_len = 4096              # 必填，正整数
seed = 42                       # 可省，默认 42

[[source]]
name = "web"                    # 必填，源标识（报告与 meta 用）
dataset = "my-web"              # 必填，convert 时的 --name 缓存别名
subset = "default"              # 可省，默认 "default"
split = "train"                 # 可省，默认 "train"
field = "text"                  # 可省，默认 "text"，要 tokenize 的字符串列
ratio = 0.7                     # 必填，> 0；各源比例只表示相对权重，不强制归一化

[[source]]
name = "code"
dataset = "my-code"
ratio = 0.3
```

校验规则：TOML 语法错误、缺必填键、`max_seq_len` 非正整数、`ratio ≤ 0`、源列表为空、源 `name` 重复 → 均在启动前报错退出并指明字段。

## 输出布局

```
<output>/
├── tokens.bin      # 扁平 token ids，按 max_seq_len 紧密切分
└── meta.json       # 元数据（schema 见下）
```

- `--output` 缺省为 `<cache-dir>/mixtures/<配置文件名去扩展名>`；已存在时报错，`--overwrite` 清空重建
- 总 token 数 = `num_sequences × max_seq_len`（尾部丢弃的不计）

### meta.json schema

```json
{
  "format": "trainops-mix/v1",
  "tokenizer": {"preset": "llama2"},
  "vocab_size": 32000,
  "dtype": "uint16",
  "max_seq_len": 4096,
  "seed": 42,
  "num_sequences": 123456,
  "total_tokens": 505971712,
  "dropped_tail_tokens": 123,
  "exhausted_sources": ["code"],
  "sources": [
    {"name": "web", "dataset": "my-web", "subset": "default", "split": "train",
     "configured_ratio": 0.7, "docs_total": 120000, "docs_sampled": 98765,
     "tokens": 354000000}
  ]
}
```

`exhausted_sources`：自然抽干而停止的源；正常跑完所有源（全部耗尽恰好同时结束属极小概率，仍记录）时为空。训练脚本据此自检：tokenizer 是否一致、长度是否符合预期、各源实际 token 占比与配置比例的偏差。

### 训练侧用法

```python
import numpy as np, json
meta = json.load(open("<output>/meta.json"))
tokens = np.memmap("<output>/tokens.bin", dtype=meta["dtype"], mode="r")
seqs = tokens.reshape(-1, meta["max_seq_len"])          # (N, L) 可直接索引
# 一条序列：input_ids = seq[:-1]，labels = seq[1:]
```

## 打包算法

**阶段一（pretokenize）**：逐源用 `DatasetCache` 定位（与 convert 完全对称：`dataset`/`subset`/`split` → `<cache-dir>/datasets/<name>/<subset>/<split>/data.arrow`），`load_from_disk` 后按 `field` 列逐篇 `encode` → 追加 EOS → 写入临时 `<tmpdir>/<source>.bin`（统一 uint32 中间格式）+ 文档偏移边车文件（uint64 偏移数组，实现细节自定）；产出该源文档总数/token 总数统计。空文档跳过计数。临时目录在运行结束或异常时清理。

**阶段二（pack）**：每个源维护"下一个文档"游标。循环：

1. 按固定权重 `ratio` 做一次加权抽样选中源（seeded `numpy.random.default_rng`，抽样只用于源选择，保证确定性）
2. 取该源下一篇文档的 token 填入 buffer；buffer ≥ `max_seq_len` 时切出完整序列写入 `tokens.bin`
3. 任一源文档取尽 → 停止；停止时先将 buffer 中完整的序列全部冲刷写入，余下不足一整序列的尾部丢弃计数

权重固定不重归一化：小源会较早抽干（预期行为），实现与"比例 = 期望占比"的语义一致。

## CLI 接口

```bash
trainops mix --config mix.toml \
             [--output /path/to/dir] [--overwrite] \
             [--cache-dir ~/.cache/trainops]
```

- `--config`：TOML 配置文件路径（必填）
- `--output`：输出目录，默认 `<cache-dir>/mixtures/<config 文件名去扩展名>`
- `--overwrite`：输出目录已存在时清空重建（无此选项则报错）
- `--cache-dir`：缓存根目录，默认 `~/.cache/trainops`
- 进度展示与 fetch/convert 一致：TTY 用 tqdm 进度条（阶段一按源、阶段二按序列数），非 TTY 定期 stderr 进度行
- 结束时打印总结：序列数、总 token 数、各源配置比例 vs 实际 token 占比、丢弃尾部长度、耗尽源（如有）

## 错误处理

- **配置非法**（语法/缺键/ratio ≤ 0/源为空/name 重复）→ exit 1，指明字段
- **Arrow 缓存不存在** → exit 1，提示先运行 convert
- **预置 tokenizer 缺依赖**（未安装 transformers）→ exit 1，提示 `pip install trainops[hf]`
- **自定义类文件不存在或接口不齐** → exit 1，指明缺失的属性/方法
- **field 列在 Arrow 中不存在** → exit 1，指明源与可用列
- **空源（0 篇文档）** → exit 1（属配置或上游问题，与自然抽干区分）
- **单篇 encode 抛异常** → exit 1，指明源与记录序号
- **输出目录已存在** → exit 1，`--overwrite` 重建
- **源自然抽干** → 非错误：停止打包、报告、exit 0（见"关键决策"）
- `cli.py` 的 `os._exit` 强制退出机制沿用

## 项目结构

```
src/trainops/
├── fetch/、convert/          # 不动
└── mix/
    ├── __init__.py
    ├── config.py         # TOML → dataclass 解析与校验（纯函数，可单测）
    ├── tokenizer.py      # 预置注册表 + 类文件加载（importlib）+ 鸭子类型接口校验
    ├── pretokenize.py    # 阶段一：Arrow → 临时 token 流 + 文档偏移 + 统计
    ├── pack.py           # 阶段二：加权不放回抽样 + 定长打包（纯逻辑，内存中可单测）
    └── command.py        # run_mix() 编排：解析配置 → 加载 tokenizer → 两阶段 → 写 meta.json
```

职责划分与 convert 一致：`config.py`/`pack.py` 无 IO 纯逻辑；`command.py` 负责编排与进度；`pretokenize.py` 持有 Arrow/临时文件 IO。

## 测试

- **config.py**：合法配置各字段默认值；缺 `max_seq_len`/源/ratio；ratio ≤ 0；name 重复；TOML 语法错误
- **tokenizer.py**：预置名解析（mock AutoTokenizer）；类文件加载与实例化；缺 `eos_token_id`/`vocab_size`/`encode` 时指明缺失项
- **pack.py**：固定 seed 两次运行逐字节一致；抽样分布随权重倾斜；抽干即停且报告正确源；尾部丢弃计数；长文档跨序列（断言拼接流等于预期）；EOS 位置正确；空文档跳过
- **command.py 集成**（临时目录构造 Arrow）：走 convert 造小缓存 → mix → 读回 `tokens.bin`/`meta.json` 验证 schema、序列数、dtype、确定性（同 seed 重跑 hash 一致）；输出已存在/`--overwrite`；缓存不存在报错
- 不新增 slow 集成测试（无网络依赖；预置 tokenizer 测试一律 mock）

## 依赖与工具链

- Python ≥ 3.10，uv 管理环境
- 新增运行时依赖：`tomli; python_version < "3.11"`
- 新增 extras：`hf = ["transformers>=4.40"]`（仅预置 tokenizer 需要；自定义类文件自带依赖）
- 其余复用：`datasets`/`pyarrow`（convert 已传递引入）、`tqdm`、`typer`
- 开发依赖：`pytest`（已有）
