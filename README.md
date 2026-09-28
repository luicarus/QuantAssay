# Quantassay

**在真实的 SGLang 服务路径上，比较量化前后的性能与质量。**

给同一份模型和一个量化版本（**GPTQ** 或 **AWQ**，均为 W4A16），本工具会：

1. 用 **LLM Compressor** 离线量化出 checkpoint；
2. 用 **SGLang** 依次服务原模型与量化模型，跑固定的请求负载；
3. 测量 **TTFT / TPOT / ITL / 输出吞吐**；
4. 在**你自己的数据集**上测量**困惑度**；
5. 给出回归结论、质量—性能取舍与限定证据的建议，输出 `report.md` / `report.html`。

两种量化方法产出**同一种格式**（对称 W4A16）并走**同一个 marlin kernel**，所以比较两种算法时，差异不会被"格式不同"或"执行路径不同"污染。

它不是一个跑分脚本 —— 每一层都有**可比性门控**和**证据纪律**：两侧服务参数不一致就不出百分比，质量没测就不谈精度，区间跨零就不下方向性结论。

---

## 它回答什么问题

> "我把这个模型量化成 4bit，**快了没有？质量掉了多少？值不值得？**"

问题看似简单，但常见的坑是：拿 Transformers 的推理速度当 SGLang 的服务性能、把两侧配置跑得不一致、或者在没有质量数据的情况下谈"无损量化"。本工具把这些都变成**会失败的检查**。

## 快速开始

```bash
git clone <repo> && cd quantassay  # 或你 clone 时用的目录名
pip install -r requirements/execution-layer.txt   # 需 NVIDIA GPU + Linux/WSL2
```

> **SGLang 不支持 Windows**，必须在 Linux 或 WSL2 内运行。仓库本身可在 Windows 编辑，但 GPU 工作要在 WSL2 执行层。

```bash
export PYTHONPATH=src

# 一条命令跑完全流程：量化 → 服务评估 → 质量评估 → 报告
python -m quantassay.gating \
  --run-dir "$HOME/quantassay-runs/my-first-run" \
  --model-dir "$HOME/models/llmcompare-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/<40位revision>" \
  --revision <40位revision> \
  --stage full
```

结束后看 `$HOME/quantassay-runs/my-first-run/report.md`。

### 选择 RMSNorm 后端

默认 `--operator-backend sglang` 使用 SGLang 内置算子。要试 Kernscope，先在同一 WSL SGLang 环境安装它，例如 `python -m pip install -e /path/to/kernscope --no-deps`（替换成实际 checkout 路径），再选择 `--operator-backend torch` 或 `--operator-backend triton`。SGLang 0.5.3 适配层由 QuantAssay 管理；Kernscope 只提供可复用算子。

在上面的完整流程命令中加入 `--operator-backend triton` 即可使用 Kernscope Triton。后端会进入 run 指纹和 serving 参数；切换后端必须使用新的 `--run-dir`，同一次 run 的 BF16 与量化侧使用同一后端。

详见 **[用户指南](docs/guide.md)**（含环境准备、自己的数据集、结果解读、常见问题）。

## 输出长什么样

```markdown
## Per-metric comparison
| metric | baseline | candidate | delta | change | direction |
|---|---:|---:|---:|---:|---|
| tpot   |    7.742 |    12.894 | +5.152 | +66.55% | lower is better |
| output_tokens_per_sec | 121.5 | 74.9 | -46.7 | -38.41% | higher is better |

## Quality
| side | documents | valid tokens | perplexity |
|---|---:|---:|---:|
| bf16 | 63/64 | 13078 | 30.1311 |
| gptq | 63/64 | 13078 | 42.1240 |

**Perplexity change:** +39.80% (paired 95% interval +36.67% … +43.14%, n=63)
```

## 用你自己的数据

质量语料是**配置项**，不是内置常量 —— 本工具就是让人在自己的领域数据上做选择：

```bash
python -m quantassay.gating --stage full ... \
  --corpus acme/domain-corpus \
  --corpus-split validation \
  --corpus-text-column content
```

语料身份进入 run 指纹，所以**换语料必须换 run 目录**，不会出现两套数据的数字被误比较。

## 文档

| 文档 | 内容 |
|---|---|
| **[用户指南](docs/guide.md)** | 安装、跑一次、用自己的数据、读结果、调参、FAQ |

> 需求文档、兼容性记录、工程笔记与代理约定（`AGENTS.md`）属于仓库本地材料，不随开源版本发布。用户需要的信息（含已知限制）都在用户指南里。

## 已验证的能力与已知限制

**已实测**（参考机器：4 GB 显存的 RTX 3050 Ti Laptop + WSL2）：

- GPTQ W4A16 离线量化 + `compressed_tensors_wna16_marlin` kernel 可用；
- BF16/GPTQ 顺序服务，四指标齐全、逐请求证据可查；
- 困惑度经 SGLang 原生 logprob 路径测量（**不用 Transformers 顶替**），配对 bootstrap 置信区间；
- 单条命令端到端，8/8 阶段，中断可续跑。

**已知限制（如实标注）**：

- 质量口径**只有困惑度**；任务准确率、EM/F1、输出一致性**均未实现**；
- 默认校准集仅 **4 条 prompt**，是可行性探针 —— 量化后的质量损失会被显著放大，不要据此判断某种量化算法"本身"的好坏；
- 服务性能存在**偶发的 run 间离散**（同一配置下某一侧 TPOT 可跳到约 3×，GPTQ 4.3 → 12.9 ms），**成因尚未定位**；
- 已实现 **GPTQ** 与 **AWQ** 两条量化路径（`--quant-method`），共用同一对称 W4A16 格式与同一个 marlin kernel，因此方法差异不被格式或执行路径混淆；FP8 / INT8 未实现。

## 环境要求

- **GPU**：实测环境为 4 GB 显存的笔记本卡（RTX 3050 Ti Laptop）+ WSL2 Ubuntu-24.04；更大的卡当然也可以，`--mem-fraction-static` 等参数按需调整。
- **推荐 Python 3.12**（3.10–3.12 未逐一验证）。
- 版本组合已在 [requirements/execution-layer.txt](requirements/execution-layer.txt) 锁定。**不要随意升级** `torch` / `sglang` / `llmcompressor` / `compressed-tensors`：这套组合是逐个排版本冲突试出来的，随意升级会退回 `ImportError`（典型症状与替代方案见[用户指南](docs/guide.md)）。

## 设计原则

- **只用真实的 SGLang 服务路径** —— 绝不用 `Transformers.generate()` 代替服务测量。
- **两侧必须可比** —— 服务参数逐字段校验，不一致拒绝出百分比。
- **缺失不填零** —— 未测量的指标标 `not_evaluated`，不猜测、不外推。
- **证据可追溯到原始记录** —— 每个数字都能回到逐请求/逐文档的 JSONL。
- **报告离线可开** —— 无 CDN、无外部依赖，文本转义。

## 许可证

## 许可证

[Apache-2.0](LICENSE)，版权归 luxing 所有（2026）。全文见仓库根目录的 `LICENSE`。
