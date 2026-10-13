# QuietBox 2 vs DGX Spark

Speed, wall-power efficiency and published quality scores for open-weight models on a Tenstorrent QuietBox 2
(2× p300c, 4 Blackhole chips) and NVIDIA DGX Spark (one and two machines).

**View it:** https://rothgar.github.io/qb2-vs-spark/

| file | what it is |
|---|---|
| `index.html` | the page: charts with the data built in |
| `qb2-vs-spark-results.csv` | one row per system, model, workload and user count: decode and total tok/s, time to first token, average wall watts, output tokens per joule |
| `aa-intelligence-index.csv` | Artificial Analysis Intelligence Index v4.3.2 per model, with the page each score came from |

## Method

- Every run used `vllm bench serve` with random prompts, a fixed output length and greedy sampling (temperature 0).
- QuietBox 2 served `tt serve` bundles. The Sparks served vLLM in NVIDIA's container; two Sparks ran one copy of a
  model over a direct QSFP link.
- Wall power came from a Shelly plug on each machine, sampled once a second and averaged over each run. Tokens per
  joule is output tokens per second divided by that average.
- Quality scores are published by [Artificial Analysis](https://artificialanalysis.ai), not measured on these
  machines.
