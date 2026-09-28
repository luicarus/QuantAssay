# Quantassay

Quantassay is a benchmarking pipeline for comparing a base model with its quantized counterpart on a real SGLang serving path.

It currently supports GPTQ and AWQ W4A16 quantization, serving benchmarks, and perplexity evaluation.

The pipeline covers:

- offline quantization with LLM Compressor;
- BF16 and quantized model serving with SGLang;
- TTFT, TPOT, ITL, request throughput, and output-token throughput;
- perplexity evaluation through SGLang logprobs;
- reproducible comparison reports in Markdown and HTML.

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

For each request, Quantassay records timestamps for request submission, first output arrival, subsequent streamed chunks, and stream completion.

| Metric | Definition |
|---|---|
| TTFT | Time from request submission to the first output chunk |
| TPOT | Average generation time per output token after the first token |
| ITL | Time between consecutive streamed output chunks |
| Output throughput | Successful output tokens divided by measured wall time |
| Request throughput | Successful requests divided by measured wall time |

Warm-up requests are executed before the timed window and excluded from throughput calculations.

Per-request measurements are stored as JSONL so aggregated metrics can be traced back to the raw records.

## Quick start

```bash
git clone https://github.com/luicarus/quantassay.git
cd quantassay

pip install -r requirements/execution-layer.txt
export PYTHONPATH=src
```

Run the full pipeline:

```bash
python -m quantassay.gating \
  --run-dir runs/my-first-run \
  --model-dir "$HOME/models/llmcompare-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision>" \
  --revision <revision> \
  --stage full
```

The final report is written to:

```text
runs/my-first-run/report.md
```

See [docs/guide.md](docs/guide.md) for environment setup, configuration, custom datasets, and result interpretation.

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

Results are only compared when the two runs satisfy the configured comparability checks.

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

For paired evaluations, Quantassay also reports a bootstrap confidence interval for the perplexity difference.

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
- The current benchmark runs requests sequentially and does not represent high-concurrency production traffic.

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
│   └── workload.py
├── gating.py       pipeline orchestration
└── quantize_worker.py
```

## License

Apache-2.0. See [LICENSE](LICENSE).
