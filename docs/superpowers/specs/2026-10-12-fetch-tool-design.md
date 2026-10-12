# trainops fetch 数据获取工具 — 设计文档

日期：2026-10-12
状态：已批准

## 背景

`llm-train-ops` 目标是一个 YAML 驱动的 LLM 训练 CI/CD 命令行工具（数据源配比加载、PyTorch 模型训练、评估、产物归档）。本期只实现其中的**数据获取工具 `fetch`**：

- 从魔塔社区（ModelScope）**流式**获取数据集并缓存到本地
- 支持按**字节大小配额**（如 `500MB`、`2GB`）限制拉取量，避免全量下载
- 缓存可**增量复用**：同一数据源再次请求更大配额时，只拉取差额部分
- 流水线（后续期实现）直接从本地缓存读取数据，fetch 本身不依赖 YAML 配置

## 关键决策（来自需求讨论）

| 决策点 | 结论 |
|---|---|
| 配额单位 | 字节大小（B/KB/MB/GB，1024 进制），不做精确 token 计数 |
| 缓存语义 | 增量复用：每数据源一份 append-only 缓存 + checkpoint 元数据 |
| 输入形态 | 独立 CLI 工具，参数指定单个数据源；不读 YAML |
| 缓存格式 | 原样保存：每行一条完整原始记录的 JSONL，保留所有字段 |
| 流式实现 | ModelScope 官方 SDK `MsDataset.load(..., streaming=True)` |

## 项目结构

```
llm-train-ops/
├── pyproject.toml          # uv 管理，依赖: modelscope, typer
├── src/trainops/
│   ├── __init__.py
│   ├── cli.py              # typer 入口，注册子命令（预留 train/eval 扩展位）
│   └── fetch/
│       ├── __init__.py
│       ├── command.py      # fetch 子命令参数解析与编排
│       ├── streamer.py     # MsDataset 流式拉取封装
│       ├── cache.py        # 缓存读写、checkpoint 元数据、一致性校验
│       └── sizeparse.py    # "500MB"/"2GB" → bytes
└── tests/
```

## CLI 接口

```bash
trainops fetch --dataset AI-ModelScope/xxx \
               [--subset default] [--split train] \
               --max-size 500MB \
               [--cache-dir ~/.cache/trainops]
```

- `--dataset`：ModelScope 数据集 ID（必填）
- `--subset`：子集名，默认 `default`
- `--split`：数据集 split，默认 `train`
- `--max-size`：目标缓存大小，支持 `B/KB/MB/GB`（1024 进制，大小写不敏感），必填
- `--cache-dir`：缓存根目录，默认 `~/.cache/trainops`

**字节数统计口径**：序列化后 JSONL 行（含换行符）的 UTF-8 字节数。

## 缓存布局

缓存键 = `dataset + subset + split`：

```
<cache-dir>/datasets/<dataset-slug>/<subset>/<split>/
├── data.jsonl    # append-only，每行一条原始记录（完整 JSON 序列化）
└── meta.json     # checkpoint 元数据
```

- `dataset-slug`：数据集 ID 中的 `/` 替换为 `__`（如 `AI-ModelScope/xxx` → `AI-ModelScope__xxx`）
- `meta.json` 内容：

```json
{
  "dataset": "AI-ModelScope/xxx",
  "subset": "default",
  "split": "train",
  "records": 12345,
  "bytes": 524288000,
  "updated_at": "2026-10-12T10:00:00Z"
}
```

## 拉取流程（增量复用）

1. 解析参数，`--max-size` 归一化为 bytes
2. 读取 `meta.json`（不存在则视为空缓存）
3. **缓存命中**：若 `meta.bytes >= 目标`，直接退出（exit 0），打印缓存路径与统计
4. **一致性校验**（见下节）后，计算差额 = 目标 − meta.bytes
5. 调用 `MsDataset.load(dataset, subset_name=subset, split=split, streaming=True)`，枚举流式记录：
   - 跳过前 `meta.records` 条（续拉）
   - 之后逐条 JSON 序列化 append 到 `data.jsonl`，累计新增字节
6. 每写入 1000 条 checkpoint 一次 `meta.json`（写临时文件后原子 rename）
7. 新增字节 ≥ 差额时停止，打印总结（总条数/总字节/缓存路径）

### 已知限制

流式模式无法真正 seek，续拉时前 N 条记录仍会流过网络（但不重复写盘）。这换取了零重复存储与实现简单；ModelScope 数据集一般有序，跳过前 N 条可保证不重复。

## 一致性与错误处理

- **崩溃恢复**：启动时比较 `data.jsonl` 实际文件大小与 `meta.json` 记录的 `bytes`：
  - 文件更大（上次崩溃发生在两次 checkpoint 之间）：按 `meta.bytes` 偏移截断文件，二者恢复一致
  - 文件更小或 meta 缺失但文件存在：视为缓存损坏，报错并提示删除该缓存目录后重试
- **不支持 streaming 的数据集**：捕获 SDK 异常，输出明确错误信息，非零码退出
- **网络中断 / Ctrl+C**：已 checkpoint 的数据保留，重新执行同一命令自动续拉
- **数据集不存在 / 鉴权失败**：透传 SDK 错误并附数据集 ID、subset、split 上下文

## 测试

- **单元测试**：大小解析（合法/非法输入）、slug 生成、meta 读写与原子 rename、截断恢复逻辑
- **核心流程测试**：mock 流式数据源（可控的记录生成器），验证：
  - 首次拉取达标即停
  - 缓存命中直接退出
  - 增量续拉只写差额、无重复记录
  - 模拟中断（写部分数据后抛异常）后重跑能正确恢复
- **集成测试**：标记 `slow`（默认跳过），真实拉取一个小型 ModelScope 数据集验证端到端

## 依赖与工具链

- Python ≥ 3.10，uv 管理环境
- 运行时依赖：`modelscope`、`typer`
- 开发依赖：`pytest`
- 入口：`pyproject.toml` 注册 `trainops = "trainops.cli:app"`
