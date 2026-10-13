# Adding results from your machine

Results from more hardware make the comparison more useful. This guide covers a Mac (for example an M5 Mac mini) and
a single Tenstorrent card; any machine that can serve an OpenAI-compatible API can be added the same way.

## What you need

- [uv](https://docs.astral.sh/uv/) and a clone of this repo.
- Enough memory for the models you want to test (the table below for Macs).
- Optional but valuable: a **Shelly Gen2 or newer smart plug** (Plug S, Plug US Gen4 and similar) between the machine
  and the wall. With one, the page shows your tokens per joule and running cost; without one, you can still send
  speed results. macOS `powermetrics` measures the chip, not the wall, so it isn't used.

## How much memory each model needs on a Mac

Each model runs as a 4-bit MLX build from [mlx-community](https://huggingface.co/mlx-community). macOS lets the GPU
use about 75% of unified memory, which has to hold the weights plus the KV cache. **Minimum memory** runs the
single-user tests; **for the full test** also covers 32 users at once and 32K-token prompts. `bench.py` skips models
that don't fit your Mac. Print this table with `./bench.py ram`.

| model | MLX build | weights | minimum memory | for the full test |
|---|---|---|---|---|
| qwen3.6-27b | `mlx-community/Qwen3.6-27B-4bit` | 16.1 GB | 32 GB | 48 GB |
| qwen3-coder-30b-a3b | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | 17.2 GB | 32 GB | 48 GB |
| devstral-small-2-24b | `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit` | 15.1 GB | 32 GB | 48 GB |
| gemma-4-31b | `mlx-community/gemma-4-31b-it-4bit` | 18.4 GB | 32 GB | 48 GB |
| muse-glimmer-30b | `mlx-community/Muse-Glimmer-30B-4bit` | 19.4 GB | 32 GB | 48 GB |
| ornith-1.5-35b-a3b | `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit` | 23.5 GB | 48 GB | 48 GB |
| nemotron-3.5-lightning-30b-a3b | `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` | 17.8 GB | 32 GB | 48 GB |
| qwen3.6-35b-a3b | `mlx-community/Qwen3.6-35B-A3B-4bit` | 20.4 GB | 36 GB | 48 GB |
| gpt-oss-20b | `mlx-community/gpt-oss-20b-MXFP4-Q4` | 11.2 GB | 24 GB | 32 GB |
| gpt-oss-120b | `mlx-community/gpt-oss-120b-MXFP4-Q4` | 62.3 GB | 96 GB | 128 GB |
| qwen3.5-122b-a10b | `mlx-community/Qwen3.5-122B-A10B-4bit` | 69.6 GB | 128 GB | 128 GB |
| llama-3.3-70b | `mlx-community/Llama-3.3-70B-Instruct-4bit` | 39.7 GB | 64 GB | 96 GB |

So a 64 GB M5 Pro Mac mini runs the full test of every model up to 35B and the single-user tests of Llama 3.3 70B,
but not gpt-oss-120b or Qwen3.5-122B; a 32 GB Mac runs the single-user tests of most 20 to 35B models; a 24 GB Mac
runs gpt-oss-20b.

## Run the tests on a Mac

1. **Describe your Mac in `bench.yaml`.** Copy the `mac` machine to a new id that names your hardware, and set its
   label and hardware line. Keep `models_from: mac` so it uses the Mac builds:

   ```yaml
     mac-m5-mini-32gb:
       label: Mac mini (M5, 32 GB)
       family: Apple
       hardware: "Apple Mac mini (M5, 10-core GPU, 32 GB unified memory)"
       launcher: mlx
       client: builtin
       models_from: mac
       remote_dir: ~/bench
       port: 8080
   ```

2. **Tell bench.py how to reach it** in `machines.yaml` (not committed; start from `machines.example.yaml`). Running
   on the Mac itself:

   ```yaml
   machines:
     mac-m5-mini-32gb:
       ssh: local
       power_plugs: [192.168.1.50]     # your Shelly plug's address, or [] without one
   ```

3. **Prepare the Mac:** plug it into wall power, quit other apps, turn off sleep while the tests run
   (`caffeinate -i` in another terminal), and make sure there's disk space for the models you'll test.

4. **Run.** Start with one model to check everything works, then the power test, then the full set if you have time:

   ```bash
   ./bench.py run mac-m5-mini-32gb -m gpt-oss-20b --dry-run    # what would run
   ./bench.py run mac-m5-mini-32gb -m gpt-oss-20b -w power     # minutes per model, plus the download
   ./bench.py run mac-m5-mini-32gb -w power                    # every model that fits
   ./bench.py run mac-m5-mini-32gb                             # every default test (hours)
   ```

   The first run of each model downloads it from Hugging Face. `mlx_lm.server` runs through `uvx`, and the client is
   `tools/loadgen.py`, which sends the same requests as `vllm bench serve` and records the same measurements.

5. **Check the results:** `./bench.py report --latest` prints a table, and `./bench.py site` rebuilds `docs/` so you
   can open `docs/index.html` and see your machine next to the others.

## Run the tests on a single Tenstorrent card

The `p150` machine in `bench.yaml` is a workstation with one p150 card. It serves models that have a p150 build
(the `p150:` entries under each model's `per_machine`); add entries for other bundles with a p150 profile
(`tt model search <name>` lists them). Put the workstation's ssh target and plug in `machines.yaml`, then run
`./bench.py run p150 -w power` and `./bench.py run p150` as above. The plug measures the whole workstation, which is
the number the page compares for every system. For a different card (p300c, n300, ...), copy the entry under a new id
and label.

## Send a pull request

Commit:

- `results/<your machine>/...`: the result JSONs, `.power.json` files and `meta.json` (logs are ignored).
- Your machine's entry in `bench.yaml`.

Don't commit `machines.yaml` or `docs/`; `docs/` is rebuilt when results are merged. In the pull request, say what
the machine is (chip, memory, cooling, OS version), which plug measured it, and anything unusual you saw.
`meta.json` already records the macOS and mlx-lm versions.

## Ground rules

- Don't change the workloads, sampling or prompt sizes: results are only comparable when everyone runs the same
  tests.
- Nothing else heavy running on the machine during a test.
- A model that fails or looks wrong is still worth reporting: open an issue with the log from
  `results/<machine>/<model>/<run>/`.
