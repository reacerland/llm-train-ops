# train 命令实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `trainops` 增加 `train` 命令:一个任务配置 `task.toml` 驱动混合数据集创建与模型训练,产出权重、loss 记录、checkpoint(可续训)与推理样本。

**Architecture:** 新增 `src/trainops/train/` 子包(config / model / data / checkpoint / loop / sample),复用 `mix` 的配置解析与 tokenizer 加载;`mix` 扩展为接受含 `[data]` 段的任务配置;CLI 增加 `train` 命令。模型按鸭子类型(`nn.Module`,`forward(input_ids) -> logits`)从用户 Python 文件动态加载。

**Tech Stack:** Python 3.10+, torch 2.x(已装,cu132), numpy, typer, pytest;新增直接依赖 `safetensors`(已在 uv.lock 中作为间接依赖存在)。

**Spec:** `docs/superpowers/specs/2026-10-13-train-command-design.md`

## Global Constraints

- 单 GPU 单进程;无 GPU 时退 CPU。CUDA 且 `torch.cuda.is_bf16_supported()` 时 bf16 autocast,否则 fp32。测试必须在没有 GPU 时也能通过(不依赖 CUDA)。
- 退出码:配置/用法错误 exit 2,运行错误 exit 1,成功 exit 0(仅 `train` 命令;`mix` 现有行为不变)。
- 任务配置内相对路径(`[data].output`、`[model].file`、`[data]` 内 tokenizer class 文件)一律相对配置文件所在目录解析;旧版 mix 顶层格式配置的路径解析行为不变。
- 错误消息用英文,风格与现有代码一致;错误类型指明缺失项。
- Python 3.10 兼容:TOML 解析沿用 `try: import tomllib / except: import tomli` 模式。
- 训练超参默认值:`batch_size` 8、`lr` 3e-4、`warmup_steps` 0、`log_interval` 10、`save_interval` 500、seed 42、`max_new_tokens` 50。
- 优化器:AdamW(betas=(0.9, 0.95), weight_decay=0.1),linear warmup + cosine decay 到 0.1×lr,grad clip 1.0。
- checkpoint / meta 类文件原子写入(临时文件 + `os.replace`),与 `_atomic_write_json` 风格一致。

## Review Focus

1. `total_tokens` 不是 `batch_size × max_seq_len` 的整数倍:step 数向上取整,始终完整 batch(多训一点,属正常)。→ Task 6 测试。
2. 模型 `logits.size(-1)` 与数据 `meta.json` 的 `vocab_size` 不一致:首个 batch 报清晰错误,而不是静默训练出垃圾。→ Task 6 测试。
3. `--resume` 时 `loss.jsonl` 必须追加续写而非清空,否则中断前的 loss 记录丢失。→ Task 6 测试。
4. 配了 `[prompts]` 但自定义 tokenizer 没有 `decode`:必须在训练开始前报错(fail fast),不能训练完几小时才失败。→ Task 7 测试。
5. `[data]` 段内 tokenizer class 文件相对路径:必须相对配置文件目录解析(CI 里 cwd 不固定)。→ Task 1 测试。

---

### Task 1: mix 接受任务配置的 `[data]` 段

**Files:**
- Modify: `src/trainops/mix/config.py`
- Modify: `src/trainops/mix/command.py`(`run_mix` 开头配置加载处)
- Test: `tests/test_mix_task_config.py`

**Interfaces:**
- Consumes: 现有 `MixConfig` / `load_config` / `TokenizerClassRef`(`mix/config.py`)。
- Produces: `resolve_mix_config(path: Path) -> tuple[MixConfig, Path | None]` —— 配置文件含 `[data]` 表时从该表解析 mix 配置并返回 `(config, output)`(`output` 已解析为绝对路径,无 `output` 字段则为 `None`);否则按现有顶层格式解析并返回 `(config, None)`。`[data]` 模式下 `TokenizerClassRef.path` 相对配置文件目录解析为绝对路径字符串。后续 Task 2 依赖此函数。

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_mix_config_task_format(tmp_path):
    tok = tmp_path / "tok.py"; tok.write_text(TINY_TOKENIZER)  # 复用 test_mix_command 的 TINY_TOKENIZER
    cfg = write_task_config(tmp_path, tok, output="rel/mixture")  # [data] 表 + [[data.source]]
    config, output = resolve_mix_config(cfg)
    assert config.max_seq_len == 4
    assert config.tokenizer.path == str(tok)                     # 已解析为绝对路径
    assert output == tmp_path / "rel" / "mixture"

def test_resolve_mix_config_legacy_format(tmp_path):
    config, output = resolve_mix_config(write_config(tmp_path, ...))  # 复用现有 write_config
    assert output is None                                             # 行为不变

def test_run_mix_uses_data_output(tmp_path):
    # [data] output="rel/mixture",不传 CLI output → 产物写到 tmp_path/rel/mixture
    result = run_mix(cfg_path, tmp_path / "cache")
    assert result.output_dir == tmp_path / "rel" / "mixture"

def test_run_mix_cli_output_overrides_config(tmp_path):
    result = run_mix(cfg_path, tmp_path / "cache", output=tmp_path / "elsewhere")
    assert result.output_dir == tmp_path / "elsewhere"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mix_task_config.py -v`
Expected: FAIL(`resolve_mix_config` 不存在 / ImportError)

- [ ] **Step 3: Implement `resolve_mix_config` and wire into `run_mix`**

`mix/config.py`:把 `load_config` 的解析主体抽为 `_parse_body(raw: dict, path: Path) -> MixConfig`(现有逻辑原样搬入);新增 `resolve_mix_config(path)`:读 TOML,含 `"data"` 键时对 `raw["data"`] 调 `_parse_body` 并解析 `output`(相对 config 父目录转绝对),同时把 `TokenizerClassRef.path` 相对 config 父目录转绝对;否则走现有路径返回 `(config, None)`,旧格式路径解析不变。`load_config` 保持原签名不变(内部调 `_parse_body`)。
`mix/command.py` `run_mix`:`config, config_output = resolve_mix_config(config_path)`;`output_dir = Path(output) if output else (config_output or cache_dir / MIXTURES_DIR_NAME / config_path.stem)`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mix_task_config.py tests/test_mix_config.py tests/test_mix_command.py -v`
Expected: PASS(新旧行为均不回归)

- [ ] **Step 5: Commit**

```bash
git add src/trainops/mix/config.py src/trainops/mix/command.py tests/test_mix_task_config.py
git commit -m "feat(mix): accept task config with [data] section and output field"
```

---

### Task 2: 任务配置解析(`train/config.py`)

**Files:**
- Create: `src/trainops/train/__init__.py`(空)
- Create: `src/trainops/train/config.py`
- Test: `tests/test_train_config.py`

**Interfaces:**
- Consumes: `resolve_mix_config`(Task 1)、`ConfigError`(`trainops.mix.config`,直接复用,不再定义新异常)。
- Produces(后续任务全部依赖这些名字):

```python
@dataclass
class ModelRef:
    file: str                    # 配置原样字符串;加载时相对 config 目录解析
    class_name: str
    args: dict[str, Any]

@dataclass
class TrainParams:
    total_tokens: int
    batch_size: int = 8
    lr: float = 3e-4
    warmup_steps: int = 0
    log_interval: int = 10
    save_interval: int = 500
    seed: int = 42

@dataclass
class PromptsConfig:
    texts: list[str]
    max_new_tokens: int = 50

@dataclass
class TaskConfig:
    path: Path                   # 配置文件路径
    data: MixConfig
    data_output: Path | None     # 来自 resolve_mix_config
    model: ModelRef
    train: TrainParams
    prompts: PromptsConfig | None

def load_task_config(path: Path) -> TaskConfig   # 缺段/缺字段/非法值 raise ConfigError
```

- [ ] **Step 1: Write the failing test**

```python
def test_load_task_config_full(tmp_path):
    cfg = load_task_config(write_full_task_toml(tmp_path))   # 含 [data]/[model]/[train]/[prompts]
    assert cfg.model.class_name == "GPT"
    assert cfg.model.args == {"vocab_size": 100, "d_model": 32}
    assert cfg.train.total_tokens == 2560
    assert cfg.train.batch_size == 8                          # 默认值生效
    assert cfg.prompts.texts == ["hello"]

def test_load_task_config_defaults_and_no_prompts(tmp_path):
    cfg = load_task_config(write_minimal_task_toml(tmp_path))  # 只含必填
    assert cfg.prompts is None
    assert cfg.train.lr == 3e-4 and cfg.train.seed == 42

def test_missing_model_section(tmp_path):
    with pytest.raises(ConfigError, match="\\[model\\]"):
        load_task_config(write_toml(tmp_path, data_only=True))

def test_missing_total_tokens(tmp_path):
    with pytest.raises(ConfigError, match="total_tokens"):
        load_task_config(write_toml(tmp_path, no_total_tokens=True))

def test_invalid_lr_type(tmp_path):  # lr = "fast"
    with pytest.raises(ConfigError):
        load_task_config(write_toml(tmp_path, lr='"fast"'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_config.py -v`
Expected: FAIL(`trainops.train.config` 不存在)

- [ ] **Step 3: Implement `load_task_config`**

`data` / `data_output` 直接调 `resolve_mix_config(path)`;`[model]` 必填 `file` / `class`(TOML 里键名 `class`,映射到 `class_name`),`[model.args]` 缺省 `{}`;`[train]` 必填 `total_tokens`(正整数),其余按 Global Constraints 默认值;`[prompts]` 可选,必填非空 `texts` 列表。所有校验失败 raise `ConfigError`,消息含配置文件路径与字段名,风格同 `mix/config.py`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trainops/train/__init__.py src/trainops/train/config.py tests/test_train_config.py
git commit -m "feat(train): task config parsing"
```

---

### Task 3: 模型加载 + GPT 测试 fixture

**Files:**
- Create: `src/trainops/train/model.py`
- Create: `tests/fixtures/__init__.py`(空)、`tests/fixtures/gpt.py`
- Test: `tests/test_train_model.py`

**Interfaces:**
- Consumes: `ModelRef`(Task 2)。
- Produces:
  - `ModelLoadError(Exception)`
  - `load_model(ref: ModelRef, config_dir: Path) -> torch.nn.Module` —— `ref.file` 相对 `config_dir` 解析;importlib 从文件导入(模块名 `trainops_user_model_<abs(hash(resolved_path))>`,与 mix tokenizer 同款);取 `ref.class_name` 类并以 `**ref.args` 实例化;校验 `isinstance(obj, torch.nn.Module)`。文件不存在 / 类不存在 / 构造失败 / 非 nn.Module 均 raise `ModelLoadError`,消息指明问题。
  - `tests/fixtures/gpt.py` 提供 `class GPT(torch.nn.Module)`,构造参数 `vocab_size, d_model, n_layer, n_head, max_seq_len, dropout=0.0`,`forward(input_ids) -> logits [B, T, vocab_size]`。Task 6/7/8 的测试都用它。

- [ ] **Step 1: Write the failing test**

```python
def test_load_model_ok(tmp_path):
    model = load_model(ModelRef(file="gpt.py", class_name="GPT",
                                args={"vocab_size": 50, "d_model": 32,
                                      "n_layer": 2, "n_head": 2, "max_seq_len": 16}),
                       config_dir=FIXTURES_DIR)
    assert isinstance(model, torch.nn.Module)
    logits = model(torch.randint(0, 50, (2, 16)))
    assert logits.shape == (2, 16, 50)

def test_load_model_file_missing():      # ModelLoadError, match="not found"
def test_load_model_class_missing():     # ModelLoadError, match=类名
def test_load_model_not_a_module():      # fixture 里一个普通类 → ModelLoadError, match="nn.Module"
def test_load_model_bad_args():          # args 缺 vocab_size → ModelLoadError 包装构造异常

@pytest.mark.slow
def test_gpt_fixture_30m_params():
    model = GPT(vocab_size=32000, d_model=512, n_layer=8, n_head=8, max_seq_len=4096)
    n = sum(p.numel() for p in model.parameters())
    assert 28_000_000 <= n <= 35_000_000
    # 一次 forward+backward 验证可训练(cuda 可用则上 cuda)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_model.py -v`
Expected: FAIL(`load_model` 不存在)

- [ ] **Step 3: Implement `load_model` 与 `tests/fixtures/gpt.py`**

GPT:标准 pre-norm decoder-only——token embedding + `n_layer` 个 block(`LayerNorm → 多头因果自注意(下三角 mask)→ 残差;LayerNorm → MLP(4×d_model,GELU)→ 残差`)+ 最终 LayerNorm + LM head(独立线性,不做 weight tying)。位置用可学习 position embedding(`max_seq_len` 长度)。不引入任何第三方模型库。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_model.py -v && pytest tests/test_train_model.py -v --runslow -k 30m`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trainops/train/model.py tests/fixtures/ tests/test_train_model.py
git commit -m "feat(train): duck-typed model loading + GPT test fixture"
```

---

### Task 4: 混合数据集加载与 batcher

**Files:**
- Create: `src/trainops/train/data.py`
- Test: `tests/test_train_data.py`

**Interfaces:**
- Consumes: mix 产物(`tokens.bin` + `meta.json`,格式 `trainops-mix/v1`)。
- Produces:

```python
class MixtureNotFoundError(Exception): ...

@dataclass
class Mixture:
    sequences: np.memmap     # shape (num_sequences, max_seq_len), dtype uint16/uint32
    vocab_size: int
    max_seq_len: int
    num_sequences: int

def load_mixture(directory: Path) -> Mixture
    # 目录或文件缺失 → MixtureNotFoundError,消息含 "run mix first";
    # meta.json format != "trainops-mix/v1" → ValueError

class Batcher:
    def __init__(self, mixture: Mixture, batch_size: int, seed: int): ...
    def next_batch(self) -> np.ndarray      # int64, shape (batch_size, max_seq_len)
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

- [ ] **Step 1: Write the failing test**

```python
def test_load_mixture_ok(tmp_path):        # 手写 tokens.bin/meta.json(6 条 seq_len=4 uint16)
    m = load_mixture(tmp_path)
    assert m.sequences.shape == (6, 4) and m.vocab_size == 100

def test_load_mixture_missing(tmp_path):
    with pytest.raises(MixtureNotFoundError, match="run mix first"):
        load_mixture(tmp_path / "nope")

def test_batcher_deterministic():          # 同 seed 两个 Batcher 逐 batch 相等
def test_batcher_epoch_wrap_covers_all():  # batch_size=2,6 条数据,3 个 batch 覆盖全部索引;第 4 个 batch 进入新 epoch
def test_batcher_state_roundtrip():        # 取 2 个 batch → state_dict → 新 Batcher load_state_dict → 后续 5 个 batch 与原 Batcher 逐一相等
def test_next_batch_dtype_int64()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_data.py -v`
Expected: FAIL(`trainops.train.data` 不存在)

- [ ] **Step 3: Implement**

`Batcher`:`np.random.default_rng(seed)`;每 epoch 开始 `rng.permutation(num_sequences)`;`next_batch` 按序取 `batch_size` 条,越界即开新 epoch;返回 `.astype(np.int64)`。`state_dict` 存 `{"epoch", "pos", "rng": rng.bit_generator.state}`;`load_state_dict` 还原后重建当前 epoch 的 permutation(注意:state 里需存当前 order 数组本身,避免重摇——直接存 `order` 与 `pos` 即可,外加 rng state 供下一 epoch)。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trainops/train/data.py tests/test_train_data.py
git commit -m "feat(train): mixture memmap loading and deterministic batcher"
```

---

### Task 5: checkpoint 保存与恢复

**Files:**
- Create: `src/trainops/train/checkpoint.py`
- Test: `tests/test_train_checkpoint.py`

**Interfaces:**
- Consumes: `Batcher`(Task 4)。
- Produces:

```python
def save_checkpoint(path: Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    step: int, batcher: Batcher) -> None
    # torch.save dict: step/model/optimizer/scheduler/batcher +
    # torch/numpy/random 的 RNG 状态(cuda 可用时含 cuda rng);
    # 原子写:先写 path.with_suffix(".pt.tmp") 再 os.replace

def load_checkpoint(path: Path, *, model, optimizer, scheduler,
                    batcher: Batcher) -> int   # 还原全部状态并返回 step
```

- [ ] **Step 1: Write the failing test**

```python
def test_checkpoint_roundtrip(tmp_path):
    model, opt, sched, batcher = make_training_state(steps=5)   # 小模型走 5 步
    save_checkpoint(tmp_path / "checkpoint.pt", model=model, optimizer=opt,
                    scheduler=sched, step=5, batcher=batcher)
    fresh = make_fresh_state()
    step = load_checkpoint(tmp_path / "checkpoint.pt", **fresh)
    assert step == 5
    assert state_dicts_equal(fresh["model"], model)
    # 继续训 3 步:fresh 与原始实例的最终参数逐元素相等(fp32 CPU)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_checkpoint.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

`load_checkpoint` 内 `torch.load(path, map_location="cpu", weights_only=False)`(checkpoint 含优化器状态与 RNG,weights_only 不适用);恢复后按当前 device 再 `to(device)` 的责任在调用方(Task 6 的 `run_training`)。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trainops/train/checkpoint.py tests/test_train_checkpoint.py
git commit -m "feat(train): atomic checkpoint save/restore with full training state"
```

---

### Task 6: 训练循环

**Files:**
- Create: `src/trainops/train/loop.py`
- Test: `tests/test_train_loop.py`

**Interfaces:**
- Consumes: `TaskConfig`(Task 2)、`load_model`(Task 3)、`load_mixture`/`Batcher`(Task 4)、`save_checkpoint`/`load_checkpoint`(Task 5)、`generate_samples`(Task 7,本任务只在 `config.prompts` 非空时调用,可用 monkeypatch 替身测试)。
- Produces:

```python
@dataclass
class TrainResult:
    output_dir: Path
    total_steps: int
    final_step: int
    final_loss: float

def run_training(config: TaskConfig, mixture_dir: Path, output_dir: Path,
                 *, resume: bool = False, progress: bool = True,
                 _step_hook: Callable[[int], None] | None = None) -> TrainResult
```

(`_step_hook` 在每个 step 完成后调用,仅供测试模拟中断。)

行为要点(Global Constraints + spec 已定的值不在此重复):
- `total_steps = ceil(total_tokens / (batch_size * max_seq_len))`,始终完整 batch
- device:cuda 可用则 cuda 否则 cpu;bf16 autocast 仅 cuda 且支持 bf16
- 每个 step:`input_ids=batch[:, :-1]`,`labels=batch[:, 1:]`,`F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))`;backward → clip 1.0 → optimizer → scheduler
- 每 `log_interval` 步:追加 `loss.jsonl` 一行 `{"step", "loss", "lr", "tokens_per_sec", "elapsed_sec"}`;`progress` 时 stdout 单行同款内容
- 每 `save_interval` 步及最后一步:`save_checkpoint(output_dir / "checkpoint.pt", ...)`
- 结束:`safetensors.torch.save_file(model.state_dict(), output_dir / "model_final.safetensors")`;`shutil.copy(config.path, output_dir / "task.toml")`
- 首个 batch 校验 `logits.size(-1) == mixture.vocab_size`,否则 ValueError 指明模型词表与数据 tokenizer 不匹配
- `resume=True`:先建模型/optimizer/scheduler/batcher,再 `load_checkpoint(output_dir / "checkpoint.pt", ...)`,model 随后 `to(device)`;checkpoint 不存在 → `FileNotFoundError` 包装为带提示的错误消息
- `config.prompts` 非空:训练前调 Task 7 的 `require_decode(tokenizer)`(fail fast),训练后调 `generate_samples(...)` 写 `samples.txt`;tokenizer 用 `load_tokenizer(config.data.tokenizer)` 加载

- [ ] **Step 1: Write the failing test**

```python
def make_task_config(tmp_path, total_tokens=..., ...)   # tiny 数据:vocab 50, seq_len 16,
                                                        # 200 条人造规律序列(如 i%47 递增)
def test_loss_decreases_and_artifacts(tmp_path):
    result = run_training(cfg, mixture_dir, out, progress=False)
    lines = read_jsonl(out / "loss.jsonl")
    assert result.total_steps == 20                        # total_tokens=20*8*16
    assert len(lines) == 2                                 # log_interval=10
    assert lines[-1]["loss"] < lines[0]["loss"]
    assert (out / "model_final.safetensors").is_file()
    assert (out / "checkpoint.pt").is_file()
    assert (out / "task.toml").is_file()

def test_total_tokens_rounds_up(tmp_path):                 # total_tokens = 8*16*20 + 1 → 21 steps
    assert run_training(cfg_plus_one, ...).total_steps == 21

def test_vocab_mismatch(tmp_path):                         # model vocab 60 vs 数据 50
    with pytest.raises(ValueError, match="vocab"):
        run_training(bad_cfg, mixture_dir, out)

def test_resume_matches_uninterrupted(tmp_path):
    full = run_training(cfg, mixture_dir, out_a, progress=False)
    def boom(step):                                        # 第 12 步后模拟中断
        if step == 12: raise RuntimeError("simulated crash")
    with pytest.raises(RuntimeError):
        run_training(cfg, mixture_dir, out_b, _step_hook=boom, progress=False)  # save_interval=6
    resumed = run_training(cfg, mixture_dir, out_b, resume=True, progress=False)
    assert resumed.final_step == full.final_step
    assert resumed.final_loss == pytest.approx(full.final_loss, rel=1e-3)
    assert len(read_jsonl(out_b / "loss.jsonl")) == len(read_jsonl(out_a / "loss.jsonl"))  # 追加未清空
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_loop.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `run_training`**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trainops/train/loop.py tests/test_train_loop.py
git commit -m "feat(train): training loop with loss log, checkpointing and resume"
```

---

### Task 7: 推理采样(生成样本)

**Files:**
- Modify: `src/trainops/mix/tokenizer.py`(`_HFTokenizerWrapper` 增加 `decode`)
- Create: `src/trainops/train/sample.py`
- Test: `tests/test_train_sample.py`

**Interfaces:**
- Consumes: `PromptsConfig`(Task 2)、`load_tokenizer` / `TokenizerLoadError`(mix)、fixture GPT(Task 3)。
- Produces:

```python
def require_decode(tokenizer: object) -> None
    # 无 callable decode → raise TokenizerLoadError 指明缺 decode(仅 [prompts] 场景要求)

def generate_samples(model: torch.nn.Module, tokenizer, prompts: PromptsConfig,
                     device: torch.device, out_path: Path) -> None
    # greedy:逐 token forward 全序列取 argmax;命中 eos_token_id 或满
    # max_new_tokens 停止;每段写 ">>> {prompt}\n{generated}\n" 到 out_path
```

Task 6 的 `run_training` 依赖这两个名字。

- [ ] **Step 1: Write the failing test**

```python
CHAR_TOKENIZER = '''class CharTok:
    eos_token_id = 0
    vocab_size = 256
    def encode(self, text): return [ord(c) % 255 + 1 for c in text]
    def decode(self, ids): return "".join(chr(i - 1) for i in ids if i > 0)
'''

def test_generate_samples(tmp_path):            # 随机权重 GPT(vocab 256),max_new_tokens=5
    generate_samples(model, tok, PromptsConfig(texts=["ab"], max_new_tokens=5), device, out)
    content = out.read_text()
    assert ">>> ab" in content and len(content.strip().splitlines()) == 2

def test_require_decode_missing():              # 无 decode 的对象 → TokenizerLoadError, match="decode"
def test_eos_stop(tmp_path):                    # 模型恒输出 eos_token_id → 生成文本为空,不超 1 步
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_sample.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

`generate_samples`:`model.eval()` + `torch.no_grad()`;每个 prompt 独立生成;输入序列长度超 `max_seq_len`(模型构造参数,从 `model` 上取不到就用 mixture 的——由调用方 Task 6 保证 prompt 编码后 + max_new_tokens 不超限时直接截断 prompt 编码到 `max_seq_len - max_new_tokens`,截断策略为实现者自由)……简化为:prompt 编码后截断到 `max_seq_len` 以内,生成时序列达 `max_seq_len` 也停。`_HFTokenizerWrapper.decode` 直接委托 `self._hf.decode(ids)`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_sample.py -v && pytest tests/test_mix_tokenizer.py -v`
Expected: PASS(wrapper 改动不回归)

- [ ] **Step 5: Commit**

```bash
git add src/trainops/mix/tokenizer.py src/trainops/train/sample.py tests/test_train_sample.py
git commit -m "feat(train): greedy sample generation with decode-capable tokenizers"
```

---

### Task 8: CLI `train` 命令 + 端到端 + 文档

**Files:**
- Modify: `src/trainops/cli.py`(新增 `train` 命令)
- Modify: `README.md`(新增 train 一节与流水线三步流程)
- Modify: `pyproject.toml`(dependencies 加 `"safetensors"`)
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: 全部前序任务;CLI 捕获 `ConfigError`(exit 2)、`ModelLoadError` / `MixtureNotFoundError` / `OutputExistsError` / `ValueError` / `FileNotFoundError`(exit 1)。
- Produces: `trainops train --config PATH --output PATH [--resume] [--overwrite] [--cache-dir PATH]`。

CLI 行为(spec 已定):`--resume` 与 `--overwrite` 同给 → exit 2;输出目录已存在且非 resume/overwrite → exit 1;`overwrite` → 清空重建;mixture 目录解析:`config.data_output or cache_dir/"mixtures"/config.path.stem`。

- [ ] **Step 1: Write the failing test(CLI 参数与退出码,monkeypatch `run_training`)**

```python
def test_train_command_success(tmp_path, monkeypatch):     # 仿照 test_fetch_command_success
    # 断言透传 config/output/resume=False,exit 0
def test_train_resume_and_overwrite_conflict(tmp_path):    # exit 2
def test_train_config_error_exit_2(tmp_path):              # 坏 toml → exit 2
def test_train_output_exists(tmp_path):                    # exit 1,提示 --overwrite/--resume
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI 命令并加 safetensors 依赖**

`cli.py` 新增 `train` 命令,错误处理块风格照抄现有命令;`pyproject.toml` dependencies 加 `"safetensors"`;`uv lock` 刷新。

- [ ] **Step 4: 端到端测试(真实训练,不 mock)**

写进 `tests/test_train_cli.py`:tmp 目录下造 char tokenizer 文件(复用 Task 7 的 CHAR_TOKENIZER)、fixture GPT 指针、`task.toml`(`[data]` 单源 + `[model]` + `[train]` total_tokens 给 ~10 步 + `[prompts]`)、用 `make_arrow_cache`(test_mix_command 的 helper 搬进 conftest 或复制)造数据;CliRunner 依次调 `mix --config task.toml` 与 `train --config task.toml --output out`;断言:exit 0、`model_final.safetensors` / `loss.jsonl` / `samples.txt` / `task.toml` 存在,`samples.txt` 含 `>>>`。

- [ ] **Step 5: Run 全部测试**

Run: `pytest -v`
Expected: PASS(全套不回归)

- [ ] **Step 6: README**

新增 `train` 一节:任务配置完整示例、三条流水线命令(fetch/convert → mix → train)、产物说明、`--resume` 用法;风格与现有章节一致(中文,命令块 + 要点列表)。

- [ ] **Step 7: Commit**

```bash
git add src/trainops/cli.py README.md pyproject.toml uv.lock tests/test_train_cli.py tests/conftest.py
git commit -m "feat(train): CLI command, end-to-end pipeline test, docs"
```

---

## Self-Review 记录

- **Spec coverage**:任务配置(2)、mix 兼容(1)、模型接口(3)、数据加载(4)、checkpoint/续训(5、6)、loss 记录(6)、safetensors 产物(6、8)、推理样本(7)、CLI 与退出码(8)、README(8)、30M GPT 验证(3 slow 测试)。覆盖完整。
- **Type consistency**:`resolve_mix_config` / `ModelRef` / `TaskConfig` / `Batcher.state_dict` / `save_checkpoint`/`load_checkpoint` / `run_training` / `require_decode` / `generate_samples` 的签名在产出方与消费方任务间一致。
- **Proportion**:正文以签名与断言为主,仅 GPT 结构与 batcher state 两处给了算法描述。
