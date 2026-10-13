# train 命令设计:CICD 模型训练阶段

日期:2026-10-13
状态:待批准

## 意图与背景

`trainops` 已有 `fetch` / `convert` / `mix` 三个命令,覆盖 LLM 训练的数据准备
管线。缺最后一步:在 CICD 中执行模型训练。

CICD 中训练有两种用途,共用同一条命令、同一台机器(单 GPU 单进程),差别只在
配置:

- **冒烟训练**:小规模、几分钟跑完,验证管线端到端可用。结果(loss 记录 +
  推理样本)由人工查看判断,确认后才继续。
- **正式训练**:完整配置,产出真正可用的模型权重。

## 流水线工作流程(用户在 CI 中执行的命令)

1. **创建训练任务** = 写一个任务配置文件 `task.toml`(包含该任务的全部配置)
2. **创建混合数据集**:`trainops mix --config task.toml`
3. **训练**:`trainops train --config task.toml --output <任务产出目录>`

训练任务引用混合数据集、模型定义 class 文件和训练 token 数;训练完成后权重
保存到任务产出目录。产出目录即 CI 产物(artifact)。

## 目标 / 非目标 / 反目标

**目标**

- 一个任务配置(TOML)包含训练任务的全部配置:数据、模型、训练超参
- 模型通过指定 Python class 文件的方式加载,鸭子类型接口,无需继承基类
- 按训练 token 数(`total_tokens`)控制训练量
- 训练过程记录 loss 到产出目录,stdout 打 CI 友好的文本进度
- 断点续训:中断后 `--resume` 从最近 checkpoint 恢复
- 训练完成后可用训练出的权重执行推理(生成文本样本),供冒烟人工判断
- 同配置重跑结果确定(固定 seed)

**非目标**

- 多卡 / 分布式训练
- 除纯文本 LLM 外的模态
- 超参搜索、实验追踪平台(wandb 等)集成

**反目标**(技术上能跑也算失败的情况)

- 需要用户改动现有 `fetch` / `convert` 命令的行为
- 模型必须继承 trainops 提供的基类才能被训练
- CI 日志里 loss 进度不可读(比如被 tqdm 刷行污染)

## 约束

- 单 GPU 单进程;无 GPU 时退到 CPU(冒烟可在 CPU 上跑)
- 精度:CUDA 且支持 bf16 时用 bf16 autocast,否则 fp32(构建者无需做成可配置)
- 依赖:torch 已装;新增 `safetensors` 依赖(见产物一节)

## 任务配置 `task.toml`

一个文件包含训练任务的全部配置:

```toml
# [data] 段:与现有 mix 配置同构,外加可选 output
[data]
tokenizer = "llama2"          # 与 mix 相同:预置名,或 { class = "path/to.py:Cls" }
max_seq_len = 4096
seed = 42
output = "data/mixture"       # 可选:mix 产物目录;缺省 <cache>/mixtures/<配置名>

[[data.source]]
name = "web"
dataset = "my-web"            # convert 时的缓存别名
ratio = 0.7

[[data.source]]
name = "code"
dataset = "my-code"
ratio = 0.3

# [model] 段:模型 python class 文件
[model]
file = "models/gpt.py"
class = "GPT"

[model.args]                  # 原样传给模型构造函数的关键字参数
vocab_size = 32000
d_model = 768
n_layer = 12
n_head = 12

# [train] 段
[train]
total_tokens = 100_000_000    # 训练多少 token(必填)
batch_size = 8
lr = 3.0e-4
warmup_steps = 200
log_interval = 10             # 多少 step 记一次 loss / 打一次进度
save_interval = 500           # 多少 step 覆盖更新一次 checkpoint
seed = 42

# [prompts] 段(可选):训练结束后用最终权重做推理
[prompts]
texts = ["今天天气", "def quicksort"]
max_new_tokens = 50           # 可选,默认 50
```

字段默认值(`batch_size` 8、`lr` 3e-4、`warmup_steps` 0、`log_interval` 10、
`save_interval` 500、两个 `seed` 42)是设计者的决定,可在实现评审时调整。

**路径解析**:配置内的相对路径(`[data].output`、`[model].file`、tokenizer
class 文件)一律相对配置文件所在目录解析(设计者的决定,CI 里任务目录
自包含)。

**mix 与 train 的产物衔接**:train 按 `[data].output`(缺省
`<cache>/mixtures/<配置名>`)定位混合数据集。流水线里不要给 mix 用命令行
`--output` 另指目录——那样 train 会找不到;要自定义位置就写
`[data].output`,mix 和 train 都认它。

## 命令

### `trainops mix --config task.toml`

`mix` 扩展为接受任务配置:配置文件含 `[data]` 表时,从该表解析 mix 配置
(含可选 `output` 字段,映射为 mix 的 `--output`);不含时按现有顶层格式解析,
行为不变。命令行 `--output` 优先级高于配置内 `output`。

### `trainops train --config task.toml --output DIR [--resume] [--overwrite]`

- `--output` 必填:任务产出目录(CI artifact 目录)
- `--cache-dir`:与 mix 相同(缺省 `~/.cache/trainops`),仅用于按缺省约定
  定位混合数据集;配了 `[data].output` 时无关
- 输出目录已存在时报错(exit 1,提示 `--overwrite` 或 `--resume`),与
  `convert` / `mix` 风格一致
- `--resume`:从产出目录中的 `checkpoint.pt` 恢复(模型 / optimizer / step /
  RNG 状态全部还原),从中断处继续;目录中没有 checkpoint 时报错;
  与 `--overwrite` 互斥,同时给出报错(exit 2)
- 错误处理与退出码风格同现有命令:配置/用法错误 exit 2,运行错误 exit 1

## 模型接口(鸭子类型,无需继承)

```python
class GPT(torch.nn.Module):
    def __init__(self, **kwargs): ...      # 参数来自 [model.args]
    def forward(self, input_ids): ...       # input_ids: LongTensor [B, T]
        # 返回 logits: FloatTensor [B, T, vocab_size]
```

- 加载方式与 mix 的自定义 tokenizer 相同:`importlib` 从文件路径导入,取类,
  用 `[model.args]` 实例化;文件不存在、类不存在、构造失败、不是
  `nn.Module` 时给出指明缺失项的错误
- loss 由训练脚本计算:`input_ids = seq[:-1]`,`labels = seq[1:]`,交叉熵
- 首个 batch 时校验 `logits.size(-1)` 与数据 `meta.json` 的 `vocab_size`
  一致,不一致报清晰错误(模型词表与数据 tokenizer 不匹配)

## 数据加载与训练循环

- 数据源 = mix 产物目录(配置 `[data].output`,缺省 `<cache>/mixtures/<配置名>`,
  与 mix 的默认输出约定一致):memmap 读 `tokens.bin`,按 `meta.json` 的
  `dtype` / `max_seq_len` reshape 为定长序列
- batch 抽样:用 `[train].seed` 初始化的 RNG 对序列索引做 shuffle,顺序消费;
  一轮(epoch)用完则重新 shuffle 进入下一轮——`total_tokens` 超过数据集
  总量时自动多 epoch(设计者的决定,冒烟的小数据集依赖这一点)
- step 数 = `ceil(total_tokens / (batch_size * max_seq_len))`;最后不足一个
  epoch 没关系,按 step 数停止
- 优化器与 schedule(设计者的决定):AdamW(β=(0.9, 0.95),weight_decay=0.1)、
  linear warmup(`warmup_steps`)+ cosine decay 到 lr 的 10%、grad clip 1.0
- 确定性:固定 torch / numpy / python random seed;CUDA 上不要求逐位复现
  (cudnn benchmark 默认行为),仅保证同机器同配置可复现

## 产物(任务产出目录内容)

```
<output>/
  model_final.safetensors   # 训练结束的最终权重(state_dict)
  loss.jsonl                # 每 log_interval 一步:{"step", "loss", "lr",
                            #   "tokens_per_sec", "elapsed_sec"}
  checkpoint.pt             # save_interval 覆盖更新:模型+optimizer+step+
                            #   RNG 状态+配置快照,供 --resume
  samples.txt               # 配了 [prompts] 才有:每段为 prompt + 生成文本
  task.toml                 # 本次任务配置的快照,保证产出可溯源
```

- stdout:每个 log_interval 打一行纯文本进度
  (`step 120/1250 loss=4.213 lr=2.9e-4 tok/s=15200`),CI 日志直接可读
- `safetensors` 作为新依赖加入 `pyproject.toml`;checkpoint.pt 用
  `torch.save`(含 optimizer 状态,safetensors 不适合)
- `--resume` 恢复时 `loss.jsonl` 追加续写(不清空);checkpoint 在最后一步
  也会保存一次,保证任何中断点之后都能续(均为设计者的决定)

## 推理(冒烟人工判断用)

- 配置 `[prompts]` 时,训练结束后加载 `model_final.safetensors`,对每个
  prompt 用 greedy 解码生成 `max_new_tokens` 个 token,写入 `samples.txt`
- tokenizer 用 `[data].tokenizer` 同一配置加载(复用 mix 的加载代码)
- 编解码接口:mix 的鸭子类型只有 `encode`;推理还需要 `decode(ids) -> str`。
  处理:预置 tokenizer(HF 包装)补 `decode`;自定义 class 文件在配置了
  `[prompts]` 时要求提供 `decode`,否则报错提示(不配 `[prompts]` 则不要求)。
  校验在训练开始前做,fail fast(设计者的决定)
- 生成时模型 `eval()` 模式、无 dropout 影响;EOS 截断:生成到
  `eos_token_id` 即停(设计者的决定)

## 测试

`tests/` 下新增(测试框架沿用 pytest):

- `fixtures/gpt.py`:约 30M 参数的极简 GPT(标准 decoder-only:token
  embedding + N 层 pre-norm transformer block + LM head),符合上面的鸭子
  类型接口,供端到端验证使用;单元测试用更小的配置(~1M)以跑得快
- 配置解析:task.toml 合法/缺字段/非法值的报错;mix 对 `[data]` 表的兼容
- 模型加载:文件缺失、类缺失、非 nn.Module、[model.args] 透传
- 数据加载:batch 形状、移位正确性、同 seed 两次迭代序列一致
- 训练循环:几十步小训练,loss.jsonl 逐行合法、loss 整体下降(人造
  模式数据上)、checkpoint 文件内容完整
- 断点续训:训练中断(跑到一半步数)后 `--resume`,最终 step 数与一次性
  跑完一致,最终 loss 接近
- 推理:tiny char-level tokenizer fixture(带 `decode`),验证 samples.txt
  生成且非空
- CLI:exit code、输出目录已存在报错、`--overwrite` / `--resume` 行为

## 交给构建者的部分

- 训练循环内部结构(模块划分、函数粒度)
- 进度日志的确切格式(保持单行纯文本即可)
- cosine decay 终点比例、grad clip 阈值等次要默认值的微调
- checkpoint 的原子写入方式(先写临时文件再 rename,与现有
  `_atomic_write_json` 风格一致)
- README 中 `train` 一节的措辞,以及流水线三步流程的文档呈现方式
