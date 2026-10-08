# QuantAssay

QuantAssay is a benchmarking pipeline for comparing a base model with its quantized counterpart on a real SGLang serving path.

It supports GPTQ and AWQ W4A16 quantization, serving benchmarks, and perplexity evaluation. A separate BF16 scheduling experiment provides concurrent traffic replay and native scheduler observations.

The project uses two repositories: **QuantAssay** owns evaluation and experiment evidence; the [dedicated SGLang fork branch](https://github.com/luicarus/sglang/tree/codex/quantassay-scheduler-v0.5.3) owns waiting-aware scheduling, based on upstream SGLang v0.5.3. The fork's contribution branches are managed independently.

The pipeline covers:

- offline quantization with LLM Compressor;
- BF16 and quantized model serving with SGLang;
- TTFT, TPOT, ITL, request throughput, and output-token throughput;
- perplexity evaluation through SGLang logprobs;
- reproducible comparison reports in Markdown and HTML.

Scheduling experiments additionally record planned and actual request arrivals, native queue times, cached prompt tokens, and sampled scheduler/KV metrics. Their reports are separate from the quantization pipeline.

The serving benchmark does not use `Transformers.generate()` as a proxy for deployment performance. Requests are sent to a live OpenAI-compatible SGLang endpoint and measured from the client side.

## Pipeline

```text
Base model
    |
    +----------------------+
    |                      |
    |                  Quantization
    |                 GPTQ / AWQ
    |                      |
    v                      v
  BF16                  W4A16
    |                      |
    +------ SGLang --------+
             |
             v
       Serving benchmark
       TTFT / TPOT / ITL
       tokens/s / requests/s
             |
             v
       Perplexity evaluation
             |
             v
       report.md / report.html
```

GPTQ and AWQ use the same symmetric W4A16 representation and the same Marlin execution path where supported. This keeps the serving path consistent when comparing the two quantization methods.

## Serving metrics

Serving measurements are collected from streamed SGLang responses.

For each request, QuantAssay records timestamps for request submission, first output arrival, subsequent streamed chunks, and stream completion.

| Metric | Definition |
|---|---|
| TTFT | Time from request submission to the first output chunk |
| TPOT | `(E2E - TTFT) / (output_tokens - 1)`; unavailable below two output tokens |
| ITL | Time between consecutive streamed output chunks |
| Output throughput | Successful output tokens divided by measured wall time |
| Request throughput | Successful requests divided by measured wall time |

Warm-up requests are executed before the timed window and excluded from throughput calculations.

Per-request measurements are stored as JSONL so aggregated metrics can be traced back to the raw records.

## Quick start

```bash
git clone https://github.com/luicarus/QuantAssay.git
cd QuantAssay

pip install -r requirements/execution-layer.txt
export PYTHONPATH=src
```

Select the maintained fork sources:

```bash
git clone --depth 1 --single-branch \
  --branch codex/quantassay-scheduler-v0.5.3 \
  https://github.com/luicarus/sglang.git "$HOME/src/quantassay-sglang"
ENGINE="$HOME/src/quantassay-sglang/python"
```

`--engine-source` selects that Python source tree while using the locked serving dependencies from the current environment. Omitting it uses the environment's installed SGLang.

Run the full pipeline:

```bash
REV=c1899de289a04d12100db370d81485cdf75e47ca
SNAP="$HOME/models/llmcompare-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/$REV"

python -m quantassay.gating \
  --run-dir "$HOME/quantassay-runs/my-first-run" \
  --model-dir "$SNAP" \
  --revision "$REV" \
  --engine-source "$ENGINE" \
  --stage full
```

Set `SNAP` to your local source-model snapshot. The commands below reuse the same `SNAP` and `REV`.

The final report is written to:

```text
$HOME/quantassay-runs/my-first-run/report.md
```

### RMSNorm backend

The default `--operator-backend sglang` uses SGLang's built-in operators. To try Kernscope, install it in the same WSL SGLang environment, for example with `python -m pip install -e /path/to/kernscope --no-deps`, then select `--operator-backend torch` or `--operator-backend triton`. QuantAssay manages the SGLang 0.5.3 adapter; Kernscope supplies reusable operators.

Add `--operator-backend triton` to the full-pipeline command above to use Kernscope Triton. The backend is part of the run fingerprint and serving parameters, so use a new `--run-dir` when switching backends and use the same backend for BF16 and quantized sides of a run.

See [docs/guide.md](docs/guide.md) for environment setup, configuration, custom datasets, and result interpretation.

## Concurrent scheduling experiments

The independent scheduling entry point currently supports SGLang 0.5.3 and Qwen3-0.6B BF16. Run the native FCFS and longest-prefix-match (LPM) policies sequentially:

```bash
python -m quantassay.scheduling \
  --run-dir "$HOME/quantassay-runs/scheduling-fcfs-01" \
  --model-dir "$SNAP" --revision "$REV" --policy fcfs --engine-source "$ENGINE"

python -m quantassay.scheduling \
  --run-dir "$HOME/quantassay-runs/scheduling-lpm-01" \
  --model-dir "$SNAP" --revision "$REV" --policy lpm --engine-source "$ENGINE" \
  --trace-file "$HOME/quantassay-runs/scheduling-fcfs-01/workload.json"
```

Defaults are 60 requests at 16 requests/s, 64 client workers, and at most 4 running server requests. The fixed seed generates shared long prefixes, independent long inputs, and independent short inputs, with output caps of 32/64/96 tokens. Arrival modes include fixed intervals, Poisson arrivals, and bursts. `--trace-file` replays saved token IDs and arrival offsets; client dispatch delay remains visible when the worker cap is reached.

Each run saves `workload.json`, `manifest.json`, client and scheduler request JSONL, sampled native metrics, server logs, `scheduling-result.json`, and `report.md`. Queue time comes from native request-time logs, not from subtracting an estimate from TTFT. Every invocation requires a new run directory.

### Experimental LPM waiting compensation

The [fork implementation](https://github.com/luicarus/sglang/blob/codex/quantassay-scheduler-v0.5.3/python/sglang/srt/managers/schedule_policy.py) adds `lpm-aging` and makes it the branch's default scheduler. Requests below the wait threshold retain LPM ordering; requests at or above it are prioritized in queue-entry order. The default threshold is 1000 ms, configurable with `--aging-threshold-ms` in the scheduling experiment or `SGLANG_LPM_MAX_WAIT_MS` when launching the fork directly. It changes priority, and does not guarantee admission or completion within one second.

Use `ENGINE` from the fork checkout above. Both policies below run from the same checkout: `lpm` follows the native branch and `lpm-aging` follows the added branch. `warm-shared` primes the shared prefix after flushing the cache; priming is excluded from the timed measurements. The [portable patch](patches/sglang-0.5.3-lpm-aging.patch) and preparation helper remain available to reproduce earlier installed-source prototypes.

```bash
for policy in lpm lpm-aging; do
  python -m quantassay.scheduling \
    --run-dir "$HOME/quantassay-runs/${policy}-warm-01" \
    --model-dir "$SNAP" --revision "$REV" --policy "$policy" \
    --engine-source "$ENGINE" --cache-start warm-shared \
    --aging-threshold-ms 1000 \
    --trace-file "$HOME/quantassay-runs/scheduling-fcfs-01/workload.json"
done
```

Model hashes, input/arrival trace, engine and controller source hashes, cache-start protocol, and other serving parameters must match when comparing policies. Engine identity includes its actual Git commit, branch, dirty state and source hashes. Pin a commit for experiments; use new run directories after source changes. Logs record whether the added branch loaded and encountered requests over the threshold.

In two alternating warm-cache comparisons on the tested 4 GB GPU, using the default synthetic load, median native queue times were:

| Request group | Native LPM | LPM with waiting compensation |
|---|---:|---:|
| Independent long inputs | 5.55–6.67 s | 2.78–2.83 s |
| Independent short inputs | 5.47–6.30 s | 3.04–3.07 s |
| Shared long prefixes | 0.90–2.08 s | about 3.02 s |

Independent requests waited less, while shared-prefix requests waited longer. Maximum overall wait decreased in those warm-cache runs, but did not improve in a supplementary cold-cache comparison. Throughput changes were inconsistent. These observations demonstrate a fairness tradeoff under an overloaded synthetic workload; they do not establish general acceleration. One cold-cache run also had anomalous decode timings and uncertain GPU-release readings; its unchanged repeat recovered, and the cause remains unresolved.

See [the user guide](docs/guide.md#实验性-lpm-等待补偿) and [fork branch notes](https://github.com/luicarus/sglang/blob/codex/quantassay-scheduler-v0.5.3/QUANTASSAY.md) for parameters, evidence files, and limitations. The change affects queue ordering; KV allocation and execution kernels retain their existing implementation. Measurements above came from the earlier installed-source prototype, not a new GPU campaign on this Git checkout.

## Example output

```text
Per-metric comparison

metric                  baseline    candidate    change
tpot                    7.742 ms    12.894 ms    +66.55%
output_tokens_per_sec   121.5       74.9         -38.41%

Quality

side    documents    valid tokens    perplexity
bf16    63/64        13078           30.1311
gptq    63/64        13078           42.1240

Perplexity change: +39.80%
paired 95% interval: +36.67% ... +43.14%
```

Quantization reports only compare results when the two runs satisfy the configured comparability checks.

## Custom evaluation data

The quality corpus is configurable:

```bash
python -m quantassay.gating --stage full ... \
  --corpus acme/domain-corpus \
  --corpus-split validation \
  --corpus-text-column content
```

Corpus information is included in the run fingerprint. Changing the evaluation corpus therefore requires a separate run directory.

## Quantization

Currently implemented:

- GPTQ W4A16
- AWQ W4A16

Both paths use symmetric 4-bit weights with 16-bit activations.

The tested serving path uses:

```text
compressed_tensors_wna16_marlin
```

FP8 and INT8 are not implemented.

## Quality evaluation

Quality evaluation currently uses perplexity only.

Perplexity is measured through the SGLang logprob path rather than a separate Transformers inference path.

For paired evaluations, QuantAssay also reports a bootstrap confidence interval for the perplexity difference.

Task-level metrics such as accuracy, exact match, F1, and output consistency are not implemented yet.

## Reproducibility

The benchmark performs several checks before reporting relative changes:

- both sides must use the same workload;
- request contents are hashed and stored;
- serving configurations must match on comparison-sensitive fields;
- warm-up requests are excluded from the timed section;
- failed and timed-out requests remain visible in the evidence;
- unavailable metrics are reported as unavailable instead of being replaced with zero.

Raw serving records and quality measurements are persisted alongside the final report.

## Tested environment

The current implementation has been tested on:

```text
GPU:        NVIDIA RTX 3050 Ti Laptop GPU, 4 GB
OS:         Ubuntu 24.04 under WSL2
Python:     3.12
Serving:    SGLang
```

The dependency versions used for the tested environment are pinned in:

```text
requirements/execution-layer.txt
```

SGLang execution requires Linux or WSL2. The repository itself can still be edited and tested at the CPU/unit-test level on other platforms.

## Known limitations

- Quality evaluation currently covers perplexity only.
- The default calibration set contains four prompts and is intended as a functional test rather than a representative quantization study.
- Serving measurements can show run-to-run variance, especially on small consumer GPUs.
- ITL is derived from streamed response chunks. A streamed chunk is not guaranteed to correspond to exactly one tokenizer token.
- The default quantization benchmark runs requests sequentially. The separate BF16 scheduling baseline supports concurrent arrival-trace replay; see [the user guide](docs/guide.md#并发调度基准bf16). Neither synthetic workload establishes production-scale performance.
- Scheduling gauges are sampled, rather than a complete trace of every GPU batch. Cache eviction counts and KV admission-block counts are currently unavailable.
- `lpm-aging` is an experimental source patch for SGLang 0.5.3. Its fairness and throughput tradeoffs depend on load and cache state; quality and deployment suitability have not been evaluated.

## Project structure

```text
src/quantassay/
├── analysis/       comparison and regression analysis
├── evaluation/     corpus and perplexity evaluation
├── experiments/    run artifacts and persistence
├── reporting/      Markdown / HTML reports
├── serving/
│   ├── benchmark.py
│   ├── evaluator.py
│   ├── scheduling.py  concurrent replay and native scheduler evidence
│   └── workload.py
├── gating.py       pipeline orchestration
├── scheduling.py   BF16 scheduling experiment CLI
├── prepare_sglang.py  isolated SGLang source patch preparation
└── quantize_worker.py
```

## License

Apache-2.0. See [LICENSE](LICENSE).
