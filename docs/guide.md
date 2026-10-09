# 用户指南

本指南面向**使用者**：装好、跑一次、用自己的数据、看懂结果。

面向运维/复现的清单式说明（阶段依赖、产物清单、指纹纪律、失败处置）属于**仓库本地文件**（`examples/RUNBOOK.md`），不随开源版本发布；本指南覆盖使用者需要的全部内容。

---

## 目录

1. [准备环境](#1-准备环境)
2. [跑第一次](#2-跑第一次)
3. [理解产物](#3-理解产物)
4. [用自己的数据集](#4-用自己的数据集)
5. [读懂结果](#5-读懂结果)
6. [常用调整](#6-常用调整)
7. [没有 GPU / 只想重算报告](#7-没有-gpu--只想重算报告)
8. [常见问题](#8-常见问题)
9. [已知限制](#9-已知限制)

---

## 1. 准备环境

### 1.1 需要什么

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，显存越大越好。**实测最小环境是 4 GB**（RTX 3050 Ti Laptop） |
| 系统 | **Linux 或 WSL2**。SGLang 不支持 Windows |
| Python | 3.12（推荐） |
| 磁盘 | 模型权重 + 量化产物 + 运行记录，建议留 5 GB 以上 |

### 1.2 安装

```bash
git clone <repo> && cd QuantAssay  # 或你 clone 时用的目录名

# 强烈建议用独立虚拟环境
python -m venv ~/venvs/llmcompare
source ~/venvs/llmcompare/bin/activate

pip install -r requirements/execution-layer.txt
pip install -e .        # 或 export PYTHONPATH=src
```

**不要升级核心依赖。** 这里锁定的 `torch` / `sglang` / `llmcompressor` / `compressed-tensors` 组合是逐个排除版本冲突后确定的。典型冲突长这样：`llmcompressor` 要求 `compressed-tensors==0.19.0` + `transformers>=5.15.0`，而 `sglang` 要求 `==0.18.0` + `==5.12.1`，两者无法共存，报 `cannot import name 'exec_jobs_dynamic'`；换到另一组版本又会在 `torch` 上死锁。随意升级就会撞上这类 `ImportError`。

### 1.3 必须设置的环境变量

```bash
export SGLANG_IS_FLASHINFER_AVAILABLE=false
```

**为什么**：FlashInfer 的即时编译对 `nvcc` 版本挑剔，本机工具链下必然编译失败。关掉它，SGLang 会改走内置的 `sgl_kernel` 备胎，功能不受影响。

### 1.4 准备模型

需要一个**固定 commit** 的本地模型快照。用 HuggingFace 缓存布局：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 若 huggingface.co 不可达

python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-0.6B', cache_dir='$HOME/models/llmcompare-cache')
print(p)
"
```

命令会打印快照路径，形如：

```
.../models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
                                                └────────── 这就是 revision ──────────┘
```

**两个参数必须一致**：`--model-dir` 指向该快照目录，`--revision` 是末级目录名。

> 用本地已有模型也可以，只要目录里有 `config.json`、权重文件和 tokenizer，且能取出一个固定 commit id 作为 `--revision`。

---

## 2. 跑第一次

```bash
export PYTHONPATH=src

RUN="$HOME/quantassay-runs/my-first-run"
SNAP="$HOME/models/llmcompare-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/<40位revision>"
REV="<40位revision>"

python -m quantassay.gating \
  --run-dir "$RUN" \
  --model-dir "$SNAP" \
  --revision "$REV" \
  --stage full
```

### RMSNorm operator backend

The default `--operator-backend sglang` uses SGLang's built-in kernels. To route Qwen3's RMSNorm paths through Kernscope, install Kernscope in the same WSL environment and choose `torch` or `triton`. QuantAssay owns the versioned SGLang adapter; Kernscope stays framework-independent:

```bash
python -m pip install -e /path/to/kernscope --no-deps  # replace with the Kernscope checkout path
python -m quantassay.gating --run-dir "$HOME/quantassay-runs/kernscope-triton" --model-dir "$SNAP" --revision "$REV" --stage full --operator-backend triton
```

The selected backend is part of the run fingerprint and serving parameters. Use a new `--run-dir` when changing it. Both model sides within a run use the same operator backend.

约 **20 分钟**（4 GB 卡，Qwen3-0.6B）。它会依次完成 8 个阶段：

```
preflight → bf16 → gptq → gptq_service → benchmark_bf16 → benchmark_gptq → quality_bf16 → quality_gptq
  环境检查   原模型服务  离线量化   量化服务确认    serving 评估（原模型）  serving 评估（量化）  质量评估……
```

**跑完看这里**：

```
$HOME/quantassay-runs/my-first-run/report.md      ← 结论与取舍
$HOME/quantassay-runs/my-first-run/report.html    ← 同上，可离线打开
```

### 只想先确认"能不能跑"

```bash
python -m quantassay.gating ... --stage all      # 约 10 分钟，只跑能力门控
```

`all` = 环境 + 原模型服务 + 量化 + 量化服务确认。它只回答"**这台机器能不能服务这个模型**"，不产出性能或质量结论。

### 中断了怎么办

**用完全相同的参数重跑即可。** 已成功并通过校验的阶段会跳过，只重做未完成的。进度记录在 `$HOME/quantassay-runs/<id>/run-status.json`。

一个例外：如果你**改了代码或换了版本**，指纹会变化，此时必须用**新的 `--run-dir`** —— 编排会明确拒绝在旧目录上续跑，而不是把两套条件的证据混在一起。

---

## 3. 理解产物

```
$HOME/quantassay-runs/my-first-run/
├── report.md / report.html        ← 先看这两个
├── regressions.json               ← 结构化结论（含可比性判定）
├── recommendations.json           ← 条件化建议
├── run-status.json                ← 每个阶段的状态与尝试次数
│
├── benchmark_bf16.json            ← serving 汇总（原模型）
├── benchmark_gptq.json            ← serving 汇总（量化）
├── quality-bf16.json              ← 质量汇总（原模型）
├── quality-gptq.json              ← 质量汇总（量化）
├── serving-{bf16,gptq}.jsonl      ← 逐请求原始记录
├── quality-{bf16,gptq}.jsonl      ← 逐文档原始记录
├── data-manifest.json             ← 语料身份与切片指纹
│
├── logs/                          ← 每次尝试的服务/量化日志
└── artifacts/gptq-w4a16/          ← 量化 checkpoint（可复用）
```

**为什么留这么多原始记录**：报告里任何一个数字都应该能追回到原始测量。想看某个请求为什么慢，就在 `serving-*.jsonl` 里按 `request_id` 查。

**其他重要规则**：

- `$HOME/quantassay-runs/<id>/` **不会被覆盖**。想重跑就换目录名。
- **存在文件 ≠ 阶段成功**。以 `run-status.json` 里该阶段的状态为准 —— 失败阶段可能留下不完整的文件。
- `artifacts/gptq-w4a16/` 只有在 manifest 和各文件 SHA256 全部校验通过后才会被复用。

---

## 4. 用自己的数据集

**质量语料是配置项，不是内置常量。** 这是本工具的核心用法之一 —— 在自己的领域数据上比较量化方案，结论才对你有意义。

```bash
python -m quantassay.gating --stage full \
  --run-dir "$HOME/quantassay-runs/my-domain" \
  --model-dir "$SNAP" --revision "$REV" \
  --corpus acme/domain-corpus \
  --corpus-split validation \
  --corpus-text-column content
```

### 参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--corpus` | HuggingFace 数据集 id | `Salesforce/wikitext` |
| `--corpus-config` | 子集/配置名（很多数据集需要） | wikitext 用 `wikitext-2-raw-v1`，其他数据集为空 |
| `--corpus-split` | 从哪个 split 取文本 | `test` |
| `--corpus-text-column` | 存文本的列名 | `text` |
| `--corpus-revision` | 钉住数据集 commit（可选） | 不钉 |
| `--corpus-endpoint` | 镜像地址 | 已配置的 HF 镜像 |
| `--corpus-min-chars` | 跳过短于该长度的文档 | `400` |

### 三条实用规则

**① 列名写错会告诉你有什么列。**

```
CorpusError: corpus 'Salesforce/wikitext' has no column 'not_a_column';
available columns: ['text']. Pass --corpus-text-column to pick the right one.
```

**② 换语料必须换 run 目录。** 语料身份（数据集、配置、split、文本列、revision）都进入 run 指纹。在同一目录里换语料会被明确拒绝：

```
ProbeError: run directory already holds quality data for 'Salesforce/wikitext' ...,
but 'acme/domain-corpus' ... was requested. Use a new run directory ...
```

这是**保护**，不是障碍：同一个 run id 下混两套数据的测量结果，比较就没意义了。

**③ 想固定文本，就加 `--corpus-revision`。** 不加也能用 —— 语料指纹仍能检测出上游数据变了，但钉住版本更稳妥，尤其你要发布结论时。

### 用本地文件

如果数据不在 HuggingFace 上，先转成 HF `datasets` 格式存到本地，或用 `datasets.load_dataset("json", data_files=...)` 自己包一层，再把 `--corpus` 指向它。要点是：**能通过 `datasets.load_dataset(corpus, config, split=...)` 加载，且有文本列。**

### 语料被过滤空了会怎样

如果你选的数据集文档都很短，会被 `--corpus-min-chars` 全部跳过，此时会**明确报错**而不是产出一个"测量了 0 篇"的伪结果：

```
CorpusError: no usable documents in 'acme/reports' split 'train': every row was
shorter than 400 characters (or the column 'content' is empty). Lower the minimum
or choose another corpus — an empty holdout cannot measure quality.
```

把 `--corpus-min-chars` 调小即可。

---

## 5. 读懂结果

### 5.1 先看两个头部字段

```markdown
- comparability: **COMPARABLE**
- quality status: **measured**
```

- **comparability** —— 两侧是否可比。`INCOMPARABLE` 时**不会给出百分比**，因为两侧条件不同，算出来的数字没有意义。原因会写在下面。
- **quality status** —— 质量是否测过。`not_evaluated` 时**不要谈精度或"无损"**。

### 5.2 serving 指标

| 指标 | 含义 | 方向 |
|---|---|---|
| `ttft` | 请求发出到首 token 的时间（ms） | 越低越好 |
| `tpot` | 每个输出 token 的平均耗时：`(E2E - TTFT) / (output_tokens - 1)` | 越低越好 |
| `itl` | 相邻输出 token 的间隔（ms） | 越低越好 |
| `output_tokens_per_sec` | **输出** token 吞吐 | 越高越好 |
| `requests_per_sec` | 请求吞吐（**不是** token 吞吐） | 越高越好 |

**注意符号方向**：延迟类"负百分比 = 更快"，吞吐类"正百分比 = 更高"。两者不能直接比大小。

**边界情况**：TPOT 在**输出不足 2 token 时标记为 `unavailable`** —— 这是定义上的边界，不是测量失败，也不算 0。同理，缺失的指标一律标为不可用，不会被填成零。

**哪些指标可信（重要）**：跨多次运行比较发现，**TTFT 的 run 间波动很大**（同配置同侧可达 ±50% 以上），而 **TPOT / ITL / 输出 tok-s 稳定在 1–2% 内**。所以：

- 判断"哪个更快"请看 **TPOT / ITL / tok-s**；
- **不要**根据单次运行的 TTFT 差异下结论 —— 它可能只是这次运行的启动抖动。
- 想减少抖动，可增加 `--stage benchmark_bf16` 的预热请求数、多跑几次取中位数。

**必读的两段附带说明**：

- **请求覆盖**：多少请求成功/失败/超时。成功率不足时，报告会判 `incomparable`。
- **输出长度与终止原因**：如果大量请求是因为撞上 `max_tokens` 而停止（而非自然结束），报告会提示 —— 此时**可能只是长度设小了**，建议加大 `max_new_tokens` 再看差距是否仍在。

### 5.3 质量指标

```markdown
| side | documents | valid tokens | perplexity |
|---|---:|---:|---:|
| bf16 | 63/64 | 13078 | 30.1311 |
| gptq | 63/64 | 13078 | 42.1247 |

**Perplexity change:** +39.80% (paired 95% interval +36.67% … +43.14%, n=63)
```

- **困惑度（PPL）**：越低越好。这里用 `exp(总NLL / 有效token)` 汇总，**不是**对各文档的 PPL 求平均（那样会算错）。
- **置信区间**：用配对 bootstrap 在同一批文档上算。**区间跨零 ⇒ 不下方向性结论** —— 差异可能只是噪声。
- **documents 63/64**：有 1 篇没算进去。原因会在该文档的 JSONL 记录里（如 `failure_reason` / `truncated_by_context`），**不是静默丢弃**。

### 5.4 建议（recommendations）

Advisor 只基于**已测的证据**给条件化建议：

- 差异在 ±1% 以内视为噪声，不会当成方向性结论；
- 涉及修改量化配方（recipe）的建议**恒为假设**，需要你自己生成新候选并重测；
- 质量未测时，所有建议都会带上 `quality_not_evaluated` 限定。

### 5.5 一个真实例子：两种量化方法的取舍

在一次实测中（4 GB 笔记本卡，Qwen3-0.6B，wikitext-2 留出集，**仅 4 条校准样本**，63 篇可评分文档、13078 个有效 token）：

| | BF16 | GPTQ W4A16 | AWQ W4A16 |
|---|---:|---:|---:|
| **困惑度** | 30.1311 | 42.1247 | **37.9930** |
| 相对 BF16 | — | +39.80% | **+26.09%** |
| 配对 95% 置信区间 | — | +36.67% … +43.14% | **+23.88% … +28.52%** |
| TPOT (ms) | 7.911 | 4.332 | 4.388 |
| 输出 tok/s | 116.3 | 193.2 | 197.2 |
| 实测 kernel | — | `compressed_tensors_wna16_marlin` | **同一个 marlin kernel** |

**两条结论：**

**① 质量上 AWQ 明显更好。** 两种量化都让困惑度变差，但 AWQ 是 **+26%**、GPTQ 是 **+40%**，**差 13.7 个百分点且置信区间不重叠**。因为两者用的是同一套对称 W4A16 格式和同一个 kernel，这个差距可以归因于**算法本身**，而不是格式或执行路径。

**② 速度上两者没有区别。** TPOT/ITL/tok-s 全部相差在 **1–2% 以内**。这在预期之内：W4A16 下两者执行的是**同一个 marlin kernel**，算法改变的是权重，不是解码成本。**所以选择依据是质量，不是速度。**

> ⚠️ **TTFT 不参与结论**。同侧跨 run 的 TTFT 波动可达 50% 以上（BF16 实测 +57%，GPTQ +54%），所以两次单测之间的 12% 差异落在噪声内。看速度差异请以 TPOT/ITL/tok-s 为准。

**两件必须注意的事**：

1. 两个数字都建立在 **只有 4 条校准样本** 之上 —— 这是可行性探针，不代表算法在正常校准集下的水平。**质量差距的方向可信，但幅度不可外推**。要下结论请把 `--quality-calibration-samples` 加到 64 以上并用你自己的数据重测。
2. 上表的 GPTQ 是**专门为这次对照重跑**的：更早的一次 GPTQ 测量落在了那个已知的 run 间离散上（异常值），拿它比会把"异常"当成"算法差异"。**质量侧不受此影响**（困惑度可跨 run 复现到小数点后三位）。

> 这些数字来自参考机器上的两次测量（每个 run 目录放一种方法）。测量产物不随仓库发布，所以请把这些数字当作"这个工具能产出什么"的示例，而不是可直接引用的基准。

**怎么自己复现这个对照**：

```bash
# 两个 run 目录，一次一种方法，各自跑完整流程
python -m quantassay.gating --run-dir "$HOME/quantassay-runs/gptq-check" --quant-method gptq --stage full ...
python -m quantassay.gating --run-dir "$HOME/quantassay-runs/awq-check"  --quant-method awq  --stage full ...
```

然后比较两份 `report.md` 的质量与 serving 两节。

---

## 6. 常用调整

### 换量化方法

```bash
--quant-method gptq     # 默认
--quant-method awq
```

两种方法产出**同一种格式**（对称 W4A16、group_size 128、pack-quantized），并且**走同一个 marlin kernel**。这是刻意的：

- 比较两种算法时，差异不会被"格式不同"或"执行路径不同"污染；
- 服务侧参数与校验完全复用，不需要为每种方法单独调 SGLang。

**换方法必须换 run 目录。** 方法名进入 run 指纹，在同一目录里换方法会被拒绝 —— 否则上一方法的产物会被覆盖、两种算法的证据混在一个 run id 下。

```bash
python -m quantassay.gating --run-dir "$HOME/quantassay-runs/awq-check" --quant-method awq --stage full ...
```

> **AWQ 的一个已知取舍**：上游 AWQ 的示例配方用**非对称**权重，但 sglang 的 kernel 分派要求**对称**权重，否则 checkpoint 无法服务。本工具因此对 AWQ 也用对称 W4A16。这牺牲了 AWQ 论文中 zero-point 补偿的一部分收益，是"能在本机服务"与"照搬上游配方"之间的取舍，已记录在产物的 `recipe.symmetric` 与 `recipe.note` 里。如果你用非对称 scheme，量化阶段就会报错而不是等到服务时才失败。

### 换个模型

```bash
--model-dir <新快照目录> --revision <新commit>
```

模型 id 当前写死为 `Qwen/Qwen3-0.6B`（在 `src/quantassay/contracts.py` 的 `MODEL_ID`）。用别的模型需要改这一处。

### 加长输出，避免被截断

负载定义在 `configs/mvp-qwen3-0p6b.yaml` 的 `workload` 块：

```yaml
workload:
  max_new_tokens: 256     # 加大这个，看差距是否仍然存在
  timed_requests: 20
  warmup_requests: 3
  concurrency: 1
  temperature: 0
```

> **请求内容本身来自代码**（`serving/workload.py`），不从 YAML 读 —— 这样两侧服务用的输入必然一致。

### 显存不够

| 参数 | 含义 | 默认 |
|---|---|---|
| `--mem-fraction-static` | 静态分配占显存比例；调小给 KV cache 留更多余量 | `0.8` |
| `--cuda-graph-max-bs` | CUDA graph 捕获的最大 batch；调小省显存 | `2` |
| `--operator-backend` | RMSNorm backend: `sglang`, Kernscope `torch`, or Kernscope `triton` | `sglang` |

```bash
--mem-fraction-static 0.7
--cuda-graph-max-bs 1
```

### 质量评估规模

| 参数 | 含义 | 默认 |
|---|---|---|
| `--quality-documents` | 留出（dev）文档数，越多越稳越慢 | `64` |
| `--quality-calibration-samples` | 划入 calibration 分割的文档数 | `4` |
| `--quality-max-tokens` | 切分留出文档时的 token 预算 | `512` |
| `--quality-context-length` | 服务侧上下文长度；超长文档按窗口切分 | `512` |

```bash
--quality-documents 128                      # 更多留出文档（更稳，更慢）
--quality-calibration-samples 64             # 更多校准（重要，见下）
--quality-context-length 1024                # 服务侧上下文长度
```

**`--quality-calibration-samples` 值得特别注意**：它决定从语料里划多少篇进入 calibration 分割。默认 4 是**可行性探针级别**，会让量化质量看起来比实际差得多。

**超长文档会怎样**：按非重叠窗口切分以适配 `--quality-context-length`；若某个窗口仍被服务端拒绝（超出上下文），该窗口**记录为被拒**（`failure_reason` / `truncated_by_context`）而不是静默丢弃。若一篇文档的所有窗口都被拒，它会计入 `documents_total` 但不计入 `documents_scored`。

### 不要加 `--disable-cuda-graph`

那是 SGLang 0.5.20 时代的临时绕行。在较新版本上没必要，而且**只给一侧加会让两侧参数不一致，报告直接判 `incomparable`**。（这个开关保留是为了兼容旧版本。）

---

## 7. 没有 GPU / 只想重算报告

改了渲染或分析代码后，**不要重跑测量** —— 重测会产生不同的数字、破坏原始证据。直接从已有的逐请求记录重建：

```bash
python -m quantassay.reanalyze "$HOME/quantassay-runs/my-first-run"
```

它从 `serving-*.jsonl` 复算指标，重写 `regressions.json`、`recommendations.json`、`report.md/html`。

它自动识别 GPTQ 或 AWQ，并在两侧 `quality-*.json` 都存在时重新附加 PPL 对比。
存在阶段记录时，先校验质量阶段的成功状态、结果校验和以及汇总一致性。
若目录中同时存在两种方法的 benchmark，使用 `--quant-method gptq` 或
`--quant-method awq` 指定候选。质量侧复用已有汇总，不重新请求模型。

---

## 8. 常见问题

**Q：报 `preflight blocked`，说可用显存/RAM 不足。**
先确认没有其他进程占用 GPU（`nvidia-smi`）。若显存充裕但仍报错，检查是否上一次运行留下了进程：`pgrep -af sglang`。

**Q：报告说 `incomparable`，没给百分比。**
看 `regressions.json` 的 `incomparable_reasons`。常见原因：两侧服务参数不一致、请求覆盖率不足（<80%）、两侧输出长度差异过大、或有一侧被截断。**这是保护机制**，说明当前数据不足以支撑比较。

**Q：质量状态是 `not_evaluated`。**
没跑 `quality_*` 阶段。用 `--stage full`，或单独跑 `--stage quality_bf16` 然后 `--stage quality_gptq`。

**Q：报 `UnboundLocalError` 之类的错。**
更新到最新代码；这类问题已修并加了测试。

**Q：`--corpus` 加载失败。**
先确认数据集 id 拼写，以及是否需要 `--corpus-config`。很多数据集有多个子集（如 `allenai/c4` 需要 `--corpus-config en`）。若有网络问题，试 `--corpus-endpoint`。

**Q：困惑度差异很大，是不是量化算法不行？**
先看校准样本数。默认只有 4 条 —— **这几乎肯定会放大质量损失**。把 `--quality-calibration-samples` 加到 64 以上，并在自己的领域数据上重测，再判断。

**Q：同一配置跑两次，serving 数字差很多。**
已知问题：服务性能存在**偶发的 run 间离散**，同一份 checkpoint、同一个 kernel、同一套参数和负载下，某一侧的 TPOT 可能跳到健康基线的约 3 倍（GPTQ 4.3 → 12.9 ms，BF16 7.9 → 23.9 ms）。**成因尚未定位**，目前按异常值处理。

**判断时必须按侧对照各自基线** —— BF16 的健康基线本来就是约 7.9 ms，所以"峰值高但 TPOT 7.85 ms"其实是正常样本。

**质量侧不受影响**：困惑度在多次独立运行中复现到小数点后三位。若单侧显存峰值超过 3500 MiB，该 run 会被自动标记提示（弱探测，不阻断），此时**重跑该侧**即可，不必改配置。

**Q：能在 Windows 上跑吗？**
GPU 部分不行，SGLang 不支持 Windows。可以在 Windows 编辑代码，在 WSL2 里执行。

---

## 9. 已知限制

如实列出，**不要超出这些边界下结论**：

| 限制 | 说明 |
|---|---|
| **质量口径只有困惑度** | 任务准确率、EM/F1、输出一致性**未实现**。困惑度不等于任务质量 |
| **默认校准集仅 4 条** | 可行性探针级别，会显著放大质量损失。做结论请加大 |
| **serving 存在偶发 run 间离散** | 同配置可差约 3×，成因未定位。质量侧不受影响 |
| **量化方法** | 已实现 **GPTQ** 与 **AWQ**（`--quant-method`）。FP8 / INT8 未实现；FP8 还需支持它的硬件 |
| **模型 id 写死** | 当前固定 `Qwen/Qwen3-0.6B`，换模型需改 `contracts.py` |
| **`final` 留出集未使用** | 目前只用 `dev` 分割。按 PRD 设计，`final` 应在方案固定后再跑 |
| **并发负载** | 默认量化基准仍为串行；独立 BF16 调度基准支持混合长度、共享前缀和并发到达重放，尚不代表生产规模 |

**证据纪律**（工具本身强制）：

- 只用真实 SGLang 服务路径测量，不用 `Transformers.generate()` 代替；
- 未测量的指标标 `not_evaluated`，**不填零、不外推**；
- 两侧服务参数不一致**拒绝出百分比**；
- 差异在 ±1% 内视为噪声；
- 每个数字都能回到逐请求/逐文档的原始 JSONL。

## 并发调度基准（BF16）

`python -m quantassay.scheduling` 是独立的 SGLang 0.5.3 / Qwen3-0.6B BF16 实验入口，用于建立原生 FCFS/LPM 调度基线。它通过原生 `/generate` 流式接口评估，在一个服务内并发请求。

在已准备好的执行环境中，使用前文的 `$SNAP`、`$REV`，依次运行：

```bash
python -m quantassay.scheduling \
  --run-dir "$HOME/quantassay-runs/scheduling-fcfs-01" \
  --model-dir "$SNAP" --revision "$REV" --policy fcfs

python -m quantassay.scheduling \
  --run-dir "$HOME/quantassay-runs/scheduling-lpm-01" \
  --model-dir "$SNAP" --revision "$REV" --policy lpm \
  --trace-file "$HOME/quantassay-runs/scheduling-fcfs-01/workload.json"
```

默认值为 60 个请求、64 个客户端 worker、服务端最多 4 个运行请求、每秒 16 个固定间隔到达、种子 42、context length 512、CUDA graph 最大 batch 4、`mem-fraction-static=0.8`。客户端 worker 上限与服务端运行请求上限是独立设置。`--arrival-mode {burst,fixed,poisson}` 支持突发、固定间隔、固定种子的泊松到达；`--requests`、`--concurrency`、`--request-rate`、`--seed` 可调整生成负载。`--trace-file` 重放已保存的 token ID 与到达时间，覆盖请求生成参数，并校验内容指纹。每次使用新 run 目录。

负载按请求数约 60% 共享长前缀、20% 独立长输入、20% 独立短输入组成；目标输入长度约 320/64 token，输出上限循环为 32/64/96 token，实际长度完整记录。显式关闭 thinking，temperature 为 0。先执行 3 个 16-token 预热请求，再清空 radix cache 后开始计时；计时过程中允许前缀复用。此负载用于性能诊断，输出是否达到上限会在结果中披露。

到达计划在发出请求前固定，不依赖前一个响应的完成时间。客户端 worker 不足时仍保留计划时间并记录 `dispatch_lag_ms`；超过 25 ms 会标记 `client_dispatch_delayed`，不能把客户端积压解释成服务端调度效果。TTFT 从实际发出请求计时，计划到达至完成的时间另列为 `arrival_to_completion_ms`。

| 文件 | 内容 |
|---|---|
| `workload.json` | 请求文本、token ID、输入哈希、输出预算、到达时间与负载指纹 |
| `manifest.json` | 模型文件哈希、引擎关键源码哈希、控制器哈希、版本与完整服务命令 |
| `requests.jsonl` | 完成即落盘的客户端逐请求延迟、实际 token 数、失败与发送延迟 |
| `scheduler-requests.jsonl` | 按请求 ID 对齐的原生排队时间、缓存 token 数、服务日志行号 |
| `scheduler-metrics.jsonl` | 每 0.2 秒采样的等待/运行请求、KV token 使用量等指标，保留 series 标签 |
| `logs/server.log` | 原生服务、batch 与请求时间日志 |
| `scheduling-result.json` / `report.md` | 整体及请求组的分布、成功数、资源数据与限制 |
| `run-status.json` | 执行结果、观测完整性与结果文件哈希 |

原生排队时间来自调度器入队到第一次 forward，**不能用 TTFT 反推**；原生时间日志以毫秒舍入。运行请求数 gauge 是采样值，不是每个 GPU batch 的完整轨迹；ITL 是流式 chunk 间隔。缓存淘汰次数、因 KV 不足未获准入的次数目前明确标为 `unavailable`。进程树 RSS 只作诊断，共享页可能被重复计算。

对照需要模型文件、负载指纹、关键引擎源码、控制器版本与其他服务参数一致，仅调度策略不同。一次对照用于建立基线，稳定结论需要独立重复与交替顺序；此实验不产生量化收益、质量或部署结论。

“共享前缀”是请求内容类别，不代表请求到达时已有长前缀命中；独立输入也可能复用短的公共 chat template。应以逐请求 `cached_tokens` 和排队时间判断实际行为。即使负载与冷启动策略一致，缓存建立的早晚仍可能改变后续调度顺序；总缓存命中量不能替代按请求组观察公平性。

### 实验性 LPM 等待补偿

调度改动现在维护在 [SGLang fork 专用分支](https://github.com/luicarus/sglang/tree/codex/quantassay-scheduler-v0.5.3)，基于上游 `v0.5.3`。QuantAssay 负责实验与观测，不内嵌整套推理框架。该分支默认使用 `lpm-aging`；提上游 PR 的分支独立维护。

```bash
git clone --depth 1 --single-branch \
  --branch codex/quantassay-scheduler-v0.5.3 \
  https://github.com/luicarus/sglang.git "$HOME/src/quantassay-sglang"
ENGINE="$HOME/src/quantassay-sglang/python"

python -m quantassay.scheduling \
  --run-dir "$HOME/quantassay-runs/fork-aging-01" \
  --model-dir "$SNAP" --revision "$REV" --policy lpm-aging \
  --engine-source "$ENGINE" --cache-start warm-shared
```

完整量化流程同样支持 `python -m quantassay.gating ... --engine-source "$ENGINE"`。
依赖沿用当前执行环境；服务子进程与控制器选择同一源码目录。源码实际版本须为 0.5.3，Git commit、分支、工作区 dirty 状态与实际源码哈希进入身份记录；版本字符串相同不代表代码相同。建议固定 commit 再测量，源码变化后使用新 run 目录。原来已安装的 SGLang 不会被重装或覆盖。

以下复制源码与应用补丁的方式用于复现旧实验；后续开发以 fork 分支为准。

仓库提供 `patches/sglang-0.5.3-lpm-aging.patch`，修改 SGLang 0.5.3 的 `SchedulePolicy` 和策略参数选项，新增 `lpm-aging`。未达到阈值时保留原生 LPM 与批内重复前缀降优先级行为；达到阈值的请求优先按当前入队时间排序。超过 128 个排队请求时沿用 FCFS 回退。阈值表示何时补偿优先级，不保证请求在阈值内获准执行；KV 分配、请求准入与模型 kernel 保持原生实现。这个策略是实验原型，效果须按负载实测。

在 Linux/WSL 的原有执行环境中，使用 GNU `patch`，复制已安装的 SGLang 源码并应用补丁：

```bash
python -m quantassay.prepare_sglang \
  --output "$HOME/quantassay-engines/sglang053-lpm-aging" \
  --patch patches/sglang-0.5.3-lpm-aging.patch
```

此命令创建独立副本并保存原始/修改后源码哈希，不改动原安装。后续 `--engine-source` 指向包含 `sglang/` 的新目录，服务进程与观测代码均使用该目录；两组对照使用同一个源码副本，`lpm` 走原生分支，`lpm-aging` 走新增分支。补丁文件也可在 SGLang 0.5.3 仓库的 `python/` 目录内应用。

```bash
python -m quantassay.scheduling \
  --run-dir "$HOME/quantassay-runs/lpm-warm-01" \
  --model-dir "$SNAP" --revision "$REV" --policy lpm \
  --engine-source "$HOME/quantassay-engines/sglang053-lpm-aging" \
  --cache-start warm-shared

python -m quantassay.scheduling \
  --run-dir "$HOME/quantassay-runs/lpm-aging-warm-01" \
  --model-dir "$SNAP" --revision "$REV" --policy lpm-aging \
  --engine-source "$HOME/quantassay-engines/sglang053-lpm-aging" \
  --cache-start warm-shared --aging-threshold-ms 1000 \
  --trace-file "$HOME/quantassay-runs/lpm-warm-01/workload.json"
```

`--cache-start` 默认 `cold`；`warm-shared` 在清缓存后额外发送一个共享长前缀请求（输出上限 16 token），保存 `cache-prime.jsonl`，响应完成后等待 0.2 秒再开始计时。实际缓存启动协议进入负载指纹，输入/到达记录另有 `traffic_fingerprint`，不能混比冷/暖缓存结果。该预置请求不计入吞吐和延迟统计。

`--aging-threshold-ms` 默认 1000，通过服务进程的 `SGLANG_LPM_MAX_WAIT_MS` 设置，进入 run 指纹。日志与结果分别记录补偿策略是否加载、是否真的遇到超阈值队列。结果包含各请求组排队时间的 p50/p95/p99/max；需要同时考察被补偿请求与共享前缀请求的代价，以及总体吞吐。
