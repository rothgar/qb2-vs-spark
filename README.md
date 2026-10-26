# QuietBox 2 vs DGX Spark

Speed, wall-power efficiency and running cost for open-weight models on Tenstorrent, NVIDIA and Apple hardware:
a Tenstorrent QuietBox 2, one and two NVIDIA DGX Sparks, and any machine you add.

**Results:** https://rothgar.github.io/qb2-vs-spark/

Every number on the page is measured with the harness in this repo, except the quality scores, which are the
published [Artificial Analysis Intelligence Index](https://artificialanalysis.ai).

## What it measures

Each model is served on a machine and sent random prompts of a fixed length with greedy sampling (temperature 0),
shaped like coding-agent traffic:

| test | prompt tokens | output tokens | users at once |
|---|---|---|---|
| `chat` | 128 | 256 | 1, 4 |
| `codegen` | 1,024 | 1,024 | 1, 4 |
| `agent-turn` | 8,192 | 512 | 1, 4 |
| `agent-long` | 32,768 | 256 | 1 |
| `users` | 1,024 | 256 | 1, 2, 4, 8, 16, 32 |
| `power` | 1,024 | 512 | 1, 16, 32 (runs only when asked for: `-w power`) |
| `prefill` | 16,384 | 256 | 1, 4, 8 (only when asked for: `-w prefill`) |
| `prefill-cached` | 16,384, all but the last 512 shared by every request | 256 | 1, 4, 8 (only when asked for) |
| `ctx-64k`, `ctx-128k` | 65,536 and 130,816 | 256 | 1 (only when asked for, on a server with 128K of context) |
| `agent-64k`, `agent-128k` | the same, all but the last 1,024 shared by every request | 256 | 1, 4 (likewise) |

For each run it records time to first token, decode speed per user, total output tokens per second and, when the
machine is plugged into a smart plug, average wall power and output tokens per joule. The `prefill` tests are for
long-prompt work (coding agents on a large codebase, research over documents): `prefill-cached` resends the same
context with a new ending each time, like an agent's next turn, so servers that cache prompt prefixes can skip most
of the prompt. The 64K and 128K tests follow [TraceLab](https://arxiv.org/abs/2606.30560)'s measurements of Claude Code and
Codex traffic, where a typical step resends about 120K tokens of history and adds under 1K. They need a server with
128K of context: `./bench.py run spark --max-model-len 131072 -w ctx-64k -w ctx-128k -w agent-64k -w agent-128k`.
A model that doesn't have the memory for that on one machine can run on a bigger one with the same build
(`./bench.py run spark-pair --models-from spark ...`), and the page says so.

Every run sends its own random prompts (a seed per test and user count), so a server with prefix caching never
answers a measured request from an earlier run's prompts.

| machine | serves models with | client |
|---|---|---|
| Tenstorrent QuietBox 2 (2× p300c, 4 Blackhole chips) | `tt serve` (Tenstorrent catalog builds and tt-model bundles) | `vllm bench serve` |
| Tenstorrent p150 (one Blackhole card) | `tt serve`, models with a p150 build (the published p150 results ran the same bundle images on Talos Linux + Kubernetes with tt-operator) | `vllm bench serve` |
| NVIDIA DGX Spark, and two Sparks over QSFP | vLLM in NVIDIA's container | `vllm bench serve` in the same container |
| Apple silicon Mac | `mlx_lm.server` with 4-bit MLX builds | `tools/loadgen.py` (vLLM has no macOS build) |
| AMD Ryzen AI Max+ 395 and other ROCm GPUs (untested) | vLLM in a ROCm container (`gpu: rocm`) | `vllm bench serve` in the same container |

## Running it

You need [uv](https://docs.astral.sh/uv/) on the machine running `bench.py`. The machines being tested need:

- **Tenstorrent:** the `tt` CLI, and a Python venv with `vllm` for the client (`bench_python` in `bench.yaml`).
- **DGX Spark:** Docker with the NVIDIA runtime.
- **AMD:** Docker, the ROCm driver, and a vLLM ROCm image that supports the GPU.
- **Mac:** uv. Models download from Hugging Face on first use.

```bash
cp machines.example.yaml machines.yaml     # how to reach your machines and their plugs (not committed)
./bench.py run spark --dry-run              # see what would run
./bench.py run spark -m gpt-oss-20b         # every default test for one model
./bench.py run spark -w power               # the power test for every model
./bench.py report --latest                  # results/summary.md
./bench.py site                             # rebuild docs/ (the published page)
```

Results land in `results/<machine>/<model>/<run>/`: one JSON per test, a `.power.json` beside it when a plug was
sampled, and `meta.json` with the software and hardware versions. Server logs stay local.

## Layout

| path | what it is |
|---|---|
| `bench.py` | the harness: `run`, `report`, `site`, `ram` |
| `bench.yaml` | machines (grouped by `family`), models with their build on each machine, and tests |
| `machines.example.yaml` | template for `machines.yaml`: ssh targets and power plugs |
| `tools/loadgen.py` | the benchmark client for machines without vLLM |
| `tools/powerlog.py` | optional continuous plug logger |
| `site/` | the page source; `./bench.py site` builds it into `docs/` |
| `docs/` | the published page and its data (`results.csv`, `quality.csv`) |
| `results/` | every run's results |

## Adding your machine

Have an M5 Mac mini, a single Tenstorrent card, or something else? See [CONTRIBUTING.md](CONTRIBUTING.md) for how
much memory each model needs on a Mac, how to add a machine, and how to send results in a pull request. Mac results
are especially wanted: see [issue #1](https://github.com/rothgar/qb2-vs-spark/issues/1), and for AMD Strix Halo
[issue #2](https://github.com/rothgar/qb2-vs-spark/issues/2).
