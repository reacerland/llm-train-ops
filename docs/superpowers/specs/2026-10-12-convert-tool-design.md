# trainops convert 缓存转 HF Arrow 工具 — 设计文档

日期：2026-10-12
状态：已批准

## 背景

`llm-train-ops` 已有 `fetch` 工具将 ModelScope 数据集以原始 JSONL 形式增量缓存到本地（见 `2026-10-12-fetch-tool-design.md`）。本期新增 **`convert` 命令**：把缓存中指定的数据集转换为 HF Arrow 格式（`datasets.Dataset.save_to_disk` 目录），供后续模型训练时 `load_from_disk` 直接加载、经 `.map()` 接 tokenizer 转为 token_ids。

## 关键决策（来自需求讨论）

| 决策点 | 结论 |
|---|---|
| 列选择 | CLI 参数 `--fields text[,title,...]` 字段白名单提取，不做全字段保留 |
| 缓存定位 | 与 `fetch` 完全对称：`--dataset`/`--subset`/`--split`/`--name`/`--cache-dir`，复用 `DatasetCache` 目录解析 |
| 输出格式 | HF datasets 标准磁盘格式：`save_to_disk` 目录（`dataset_info.json` + Arrow 分片），`load_from_disk` 加载 |
| 缺字段行为 | 默认报错退出（指明记录序号与缺失字段）；`--skip-bad` 跳过并统计 |
| 字段值类型 | 方案 C：字符串原样保留；非字符串（数字/布尔/数组/对象）`json.dumps(ensure_ascii=False)` 序列化为 JSON 字符串；所有输出列为 Arrow `string` 类型 |
| 输出路径 | 默认与缓存并排：`<cache-dir>/datasets/<name>/<subset>/<split>/data.arrow` 目录；`--output` 可覆盖 |
| 覆盖语义 | 目标已存在默认报错；`--overwrite` 清空重建 |
| 内存模型 | 分批流式转换（每批 1000 条，RecordBatch 增量写入），内存占用恒定 |

## CLI 接口

```bash
trainops convert --dataset AI-ModelScope/xxx \
                 [--subset default] [--split train] \
                 [--name my-alias] \
                 --fields text[,title,...] \
                 [--skip-bad] \
                 [--output /path/to/dir] [--overwrite] \
                 [--cache-dir ~/.cache/trainops]
```

- `--dataset`：ModelScope 数据集 ID（必填）
- `--subset`：子集名，默认 `default`
- `--split`：数据集 split，默认 `train`
- `--name`：缓存别名（fetch 时通过 `--name` 指定的缓存键）
- `--fields`：必填，逗号分隔的字段白名单（至少一个字段）
- `--skip-bad`：记录缺字段或 JSONL 行非法时跳过并计数；默认报错退出
- `--output`：输出目录，默认 `<cache-dir>/datasets/<name>/<subset>/<split>/data.arrow`
- `--overwrite`：输出目录已存在时清空重建（无此选项则报错）
- `--cache-dir`：缓存根目录，默认 `~/.cache/trainops`

## 输出布局

```
<cache-dir>/datasets/<name>/<subset>/<split>/
├── data.jsonl      # 原缓存（fetch 写入）
├── meta.json       # fetch 的 checkpoint 元数据
└── data.arrow/     # HF datasets 目录（convert 写入，load_from_disk(".../data.arrow") 加载）
    ├── dataset_info.json
    └── data-00000-of-00001.arrow
```

目录名沿用 `data` 基名、`.arrow` 扩展名，与 `data.jsonl` 并排，表示同一数据的箭头格式版本。

## 转换语义

- **字段提取**：每条 JSONL 记录只保留 `--fields` 白名单内的字段
- **类型规则**：
  - 字符串值：原样保留
  - 非字符串值（数字、布尔、数组、嵌套对象）：`json.dumps(value, ensure_ascii=False)` 序列化为 JSON 字符串
  - JSON `null`：映射为 Arrow null，保留（不算缺字段）
  - 结果：所有输出列统一为 Arrow `string` 类型
- **缺字段**：记录缺少白名单中任一字段时——默认报错退出（exit 1，指明第几条记录缺哪个字段）；`--skip-bad` 时跳过该记录，结束时打印跳过统计
- **非法 JSON 行**：某行无法解析为 JSON 时，按坏记录处理（同缺字段的默认报错 / `--skip-bad` 跳过语义）

## 项目结构

```
src/trainops/
├── cli.py                 # 注册 convert 子命令
├── fetch/                 # 不动
└── convert/
    ├── __init__.py
    ├── command.py         # run_convert() 编排：定位缓存 → 校验 → 流式转换 → save_to_disk
    └── transform.py       # 纯函数：JSONL 行 → 提取字段 + 类型规范化 → dict[str, str|None]
```

职责划分：

- `transform.py`：无 IO 的纯转换逻辑。核心函数接收解析后的 `dict` 与字段白名单，返回 `dict[str, str | None]`；缺字段抛自定义 `MissingFieldError`，方便单测
- `command.py`：分批读取 `data.jsonl`（每批 1000 条），逐批构造 Arrow RecordBatch 增量写入 Arrow IPC 文件（内存占用恒定），完成后以内存映射方式包装为 `Dataset` 并 `save_to_disk` 到目标目录；进度展示与 fetch 一致（TTY 用 tqdm 字节进度条，非 TTY 定期 stderr 进度行）
- `fetch/cache.py` 的 `DatasetCache` 增加 `iter_records()` 方法（逐行 yield 解析后的 dict），供 convert 复用；不改 `fetch` 既有行为

## 错误处理

- **缓存不存在**：`data.jsonl` 缺失时明确报错（提示先运行 fetch），exit 1
- **缓存不一致**：复用 `ensure_consistent` 与 `CacheCorruptedError` 路径，exit 1
- **缺字段 / 非法 JSON 行**：默认报错退出（exit 1，含记录序号与原因）；`--skip-bad` 跳过并计数
- **输出目录已存在**：无 `--overwrite` 时明确报错提示，exit 1
- **转换成功**：打印总结——记录数、跳过数（如有）、输出路径、输出字节数

## 测试

- **transform.py 单元测试**：字符串保留、非字符串 JSON 序列化（数字/布尔/数组/嵌套对象）、null 保留、缺字段抛 `MissingFieldError`、unicode 不转义（`ensure_ascii=False`）
- **command.py 核心流程**（临时目录构造 JSONL 缓存）：
  - 正常转换：`load_from_disk` 读回验证字段、条数、string 列类型
  - 缺字段报错 / `--skip-bad` 跳过统计
  - 非法 JSON 行：默认报错 / `--skip-bad` 跳过
  - 输出已存在报错 / `--overwrite` 重建
  - 通过 `DatasetCache` 造缓存再 convert，验证与 fetch 缓存目录布局的集成
- 不新增 slow 集成测试（无网络依赖）

## 依赖与工具链

- Python ≥ 3.10，uv 管理环境
- 运行时依赖：无新增。`modelscope[datasets]` 已传递引入 `datasets` + `pyarrow`，`save_to_disk`/`load_from_disk` 直接用
- `cli.py` 的 `os._exit` 强制退出机制沿用（避开 datasets/pyarrow 后台线程导致的解释器关闭挂起）
- 开发依赖：`pytest`（已有）
