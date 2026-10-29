#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6"]
# ///
"""Serve each model on a machine, benchmark it, record wall power, and build the comparison page.

    ./bench.py run qb2                        # every enabled model x every default workload
    ./bench.py run mac -m gpt-oss-20b -w power
    ./bench.py run qb2 --dry-run              # print the plan and commands only
    ./bench.py report                         # results/summary.{csv,md}
    ./bench.py site                           # docs/ (the published page and its data)
    ./bench.py ram                            # unified memory each model needs on a Mac

Models, workloads and hardware descriptions live in bench.yaml; how to reach your machines (ssh targets, power
plugs) lives in machines.yaml, which is not committed (start from machines.example.yaml).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    """bench.yaml, with machines.yaml (connection details) merged over its `machines`."""
    cfg = yaml.safe_load(path.read_text())
    local = path.parent / "machines.yaml"
    if local.exists():
        for name, extra in (yaml.safe_load(local.read_text()) or {}).get("machines", {}).items():
            cfg["machines"][name] = {**cfg["machines"].get(name, {}), **extra}
    return cfg


def machine_models(cfg: dict, machine: str, models_from: str | None = None) -> list[tuple[dict, dict]]:
    """(model, per-machine entry) for every model this machine can serve. A machine with `models_from: <key>`
    (e.g. every Mac uses the `mac` entries) reads that key's entries instead of its own name; so does
    `run --models-from` (the Spark's builds on two Sparks, when one hasn't the memory for a long context)."""
    key = models_from or cfg["machines"][machine].get("models_from", machine)
    return [(m, m["per_machine"][key]) for m in cfg["models"] if key in m.get("per_machine", {})]


PRIVATE_KEYS = {"board_id", "hostname", "serial", "serial_number", "uuid"}


def scrub(x):
    """Drop identifying fields (board serials, hostnames) from environment info before it is saved."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k.lower() not in PRIVATE_KEYS}
    return [scrub(v) for v in x] if isinstance(x, list) else x


# ---------------------------------------------------------------------------------------------------- remote

class Remote:
    """Runs commands on a machine over ssh, or on this one when `ssh: local`."""

    def __init__(self, host: str, dry_run: bool = False):
        self.host, self.dry_run = host, dry_run
        self.local = host in ("local", "localhost")

    def run(self, cmd: str, *, timeout: float | None = None, check: bool = True,
            env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
        line = (prefix + " " if prefix else "") + cmd
        if self.dry_run:
            log(f"DRY {self.host}: {line}")
            return subprocess.CompletedProcess([], 0, "", "")
        argv = (["bash", "-lc", line] if self.local else
                ["ssh", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10", self.host,
                 f"bash -lc {shlex.quote(line)}"])
        return subprocess.run(argv, text=True, timeout=timeout, check=check,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def push(self, local: Path, remote_dir: str) -> None:
        if self.dry_run:
            return
        if self.local:
            dest = Path(remote_dir).expanduser()
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(local, dest / local.name)
        else:
            subprocess.run(["scp", "-q", str(local), f"{self.host}:{remote_dir}/"], check=True)

    def pull(self, remote_path: str, local: Path) -> None:
        if self.dry_run:
            return
        local.mkdir(parents=True, exist_ok=True)
        src = str(Path(remote_path).expanduser()) if self.local else f"{self.host}:{remote_path}"
        subprocess.run(["rsync", "-a", f"{src}/", f"{local}/"], stderr=subprocess.DEVNULL, check=False)


# ---------------------------------------------------------------------------------------------------- power

class PowerMeter:
    """Samples Shelly Gen2+ plugs (`power_plugs`, summed) once a second while a workload runs."""

    def __init__(self, plugs: list[str]):
        self.plugs, self.samples, self._stop = plugs, [], threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def read(plug: str) -> float:
        with urllib.request.urlopen(f"http://{plug}/rpc/Switch.GetStatus?id=0", timeout=2) as r:
            return float(json.load(r)["apower"])

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            try:
                self.samples.append((round(t, 1), round(sum(self.read(p) for p in self.plugs), 1)))
            except Exception:
                pass                                        # a missed sample is fine; the mean uses the rest
            self._stop.wait(max(0.0, 1.0 - (time.time() - t)))

    def __enter__(self) -> "PowerMeter":
        if self.plugs:
            self.start = time.time()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join()
            self.end = time.time()

    def save(self, path: Path) -> None:
        if self.plugs:
            path.write_text(json.dumps({"source": "shelly", "plugs": len(self.plugs),
                                        "window": [round(self.start, 1), round(self.end, 1)],
                                        "samples": self.samples}))


def wall_watts(result: dict, power_file: Path) -> float | None:
    """Mean wall power while the benchmark itself ran: the samples in the last `duration` seconds of the window
    (the window also covers the client starting up)."""
    if not power_file.exists():
        return None
    p = json.loads(power_file.read_text())
    if p.get("wall_w") is not None:                         # imported results carry the mean directly
        return p["wall_w"]
    end, dur = p["window"][1], result.get("duration") or 0
    ws = [w for t, w in p["samples"] if end - dur - 1 <= t <= end]
    return round(sum(ws) / len(ws), 1) if ws and len(ws) >= 0.5 * dur else None


# ---------------------------------------------------------------------------------------------------- launchers

class TTLauncher:
    """Tenstorrent: `tt serve` (catalog models and tt-model bundles), `tt model stop`."""

    def __init__(self, remote: Remote, mc: dict):
        self.r, self.mc, self.port = remote, mc, mc["port"]

    def setup(self) -> None:
        self.r.run(f"mkdir -p {self.mc['remote_dir']}")
        self.r.push(ROOT / "tools" / "fix-weight-links.sh", self.mc["remote_dir"])

    def running(self) -> list[str]:
        out = self.r.run("tt --no-color model ps --names", check=False).stdout or ""
        return [ln.strip() for ln in out.splitlines() if ln.strip()]

    def environment(self) -> dict:
        info = {}
        for key, cmd in {"tt_version": "tt --version", "device_info": "tt --no-color --json device info",
                         "kmd": "modinfo -F version tenstorrent"}.items():
            text = (self.r.run(cmd, check=False).stdout or "").strip()
            try:
                info[key] = json.loads(text)
            except json.JSONDecodeError:
                info[key] = text
        return info

    def prefetch(self, m: dict, model: dict, timeout: float) -> None:
        # tt-inference-server mounts only hub/models--org--name into its container, but the downloader leaves that
        # dir's blobs as symlinks into the shared hub/blobs store, which dangle inside the container: pull first,
        # then turn those symlinks into hardlinks
        if m.get("source") != "tt-inference-server":
            return
        self.r.run(f"tt --no-color model pull {shlex.quote(m['serve'])}", timeout=timeout, check=False)
        self.repair(model)

    def repair(self, model: dict) -> None:
        repo_dir = "~/.cache/huggingface/hub/models--" + model["hf_id"].replace("/", "--")
        self.r.run(f"{self.mc['remote_dir']}/fix-weight-links.sh {repo_dir}", check=False)

    def start(self, m: dict, timeout: float, log_path: Path) -> None:
        # each model gets a freshly reset mesh: a server that did not exit cleanly can leave an ethernet core down
        self.r.run("tt --no-color device reset --yes", check=False, timeout=300)
        cmd = ["tt", "--no-color", "serve", m["serve"], "--port", str(self.port)]
        if "/" in m["serve"]:                               # tt-model bundle
            cmd += ["--yes", "--no-update-check"] + (["--profile", m["profile"]] if m.get("profile") else [])
        res = self.r.run(" ".join(map(shlex.quote, cmd)), timeout=timeout, check=False,
                         env={"TT_SERVE_READY_TIMEOUT": str(int(timeout))})
        if not self.r.dry_run:
            log_path.write_text(res.stdout or "")
        if res.returncode != 0:
            raise RuntimeError(f"tt serve exited {res.returncode}; see {log_path}")
        # tt serve can return before the endpoint answers: poll, failing early if the server is gone
        wait_ready(self, timeout, log_path)

    def logs(self, m: dict) -> str:
        # `tt model logs` has no --tail; read the newest tt-model container's log directly
        return self.r.run("c=$(docker ps -a -q --filter name=^tt-model- | head -1); "
                          "[ -z \"$c\" ] || docker logs --tail 3000 \"$c\" 2>&1", check=False).stdout or ""

    def stop(self, m: dict) -> None:
        # bundles must be stopped by bundle id; `tt model ps` may show a different display name
        self.r.run(f"tt --no-color model stop {shlex.quote(m['serve'])}", check=False, timeout=900)
        if not self.r.dry_run and self.running():
            log(f"   !! still serving after stop: {self.running()}")


class VLLMDockerLauncher:
    """vLLM in a container (NVIDIA's on the DGX Spark): `vllm serve` in a container named bench-vllm, or, with
    `workers`, one server spanning several machines with vLLM's own multi-node launch (rank 0 serves, the others run
    headless) over a direct link (node addresses `node_ips`, head first; NCCL over RoCE on `nccl_hca`, sockets on
    `node_if`). The vllm client runs inside the same container, so hosts need only Docker and the GPU runtime: NVIDIA's
    by default, or AMD's with `gpu: rocm` (the GPU device files instead of `--gpus all`; `image` must be a vLLM ROCm
    build for that GPU). Per model: `serve`, `args` (extra `vllm serve` flags), optional `image`, `env`, `download_exclude`."""

    NAME = "bench-vllm"

    def __init__(self, remote: Remote, mc: dict):
        self.r, self.mc, self.port = remote, mc, mc["port"]
        self.workers = [Remote(h, remote.dry_run) for h in mc.get("workers", [])]
        self.nodes = [self.r, *self.workers]

    def setup(self) -> None:
        for r in self.nodes:
            r.run(f"mkdir -p {self.mc['remote_dir']}")
        self.r.push(ROOT / "tools" / "vllm-client.sh", self.mc["remote_dir"])

    def _image(self, m: dict) -> str:
        return m.get("image", self.mc["image"])

    def _gpu_flags(self) -> str:
        """Container GPU access, spelled for the remote shell. ROCm needs the device files, and the host's video and
        render groups by number (the container runs as the host user and may not know the group names)."""
        if self.mc.get("gpu", "nvidia") == "rocm":
            return ("--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined "
                    "$(for g in video render; do getent group $g | cut -d: -f3 | sed 's/^/--group-add /'; done)")
        return "--gpus all"

    def _token_flag(self) -> str:
        f = self.mc.get("hf_token_file", "~/.cache/huggingface/token")
        return f'-e HF_TOKEN="$(cat {f} 2>/dev/null)"'

    def _run_container(self, r: Remote, m: dict, cmd: str, node_ip: str | None = None) -> None:
        mc, d = self.mc, self.mc["remote_dir"]
        env = {"HF_HOME": "/hf", "HOME": "/cache-home", "VLLM_CACHE_ROOT": "/cache-home/vllm"}
        if node_ip:                                         # multi-node: all traffic between nodes on the direct link
            iface = mc["node_if"]
            env |= {"VLLM_HOST_IP": node_ip, "NCCL_SOCKET_IFNAME": iface, "GLOO_SOCKET_IFNAME": iface,
                    "TP_SOCKET_IFNAME": iface, "UCX_NET_DEVICES": iface, "NCCL_IB_HCA": mc["nccl_hca"]}
        env |= m.get("env", {})
        argv = ["--name", self.NAME, "--ipc", "host", "--network", "host",
                "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "-v", f"{d}:{d}"]
        argv += ["--device", "/dev/infiniband", "--cap-add", "IPC_LOCK"] if node_ip else []
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        argv += ["--entrypoint", "bash", self._image(m), "-c", cmd]
        # user, cache mounts and token are spelled for the remote shell ($(id -u), $HOME, $(cat token))
        shell = ("docker run -d --user $(id -u):$(id -g) " + self._gpu_flags() + " -v $HOME/.cache/huggingface:/hf "
                 f"-v $HOME/.cache/bench-home:/cache-home {self._token_flag()} ")
        r.run(f"mkdir -p ~/.cache/huggingface ~/.cache/bench-home {shlex.quote(d)} && "
              + shell + " ".join(map(shlex.quote, argv)), timeout=600)

    def running(self) -> list[str]:
        out = []
        for r in self.nodes:
            res = r.run(f"docker ps --format '{{{{.Names}}}}' --filter name=^{self.NAME}$", check=False)
            out += [f"node{i}:{ln.strip()}" for i, ln in enumerate((res.stdout or "").splitlines()) if ln.strip()]
        return out

    def environment(self) -> dict:
        info = {}
        for i, r in enumerate(self.nodes):
            gpu = ("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader" if self.mc.get("gpu", "nvidia") != "rocm"
                   else "amd-smi static --asic --driver 2>/dev/null | grep -E 'MARKET_NAME|TARGET_GRAPHICS_VERSION|DRIVER_VERSION' "
                        "|| rocm-smi --showproductname --showdriverversion 2>/dev/null")
            res = r.run(gpu + "; uname -r; "
                        "grep -h SWBUILD_VERSION /etc/dgx-release 2>/dev/null; "
                        f"docker image inspect {shlex.quote(self.mc['image'])} --format '{{{{index .RepoDigests 0}}}}'",
                        check=False)
            info[f"node{i}"] = (res.stdout or "").strip()
        return info

    def repair(self, model: dict) -> None:
        pass

    def prefetch(self, m: dict, model: dict, timeout: float) -> None:
        """Every node loads its shards from its own HF cache: download on all of them, in parallel."""
        cmd = (f"mkdir -p ~/.cache/huggingface && docker run --rm --user $(id -u):$(id -g) -e HF_HOME=/hf "
               f"{self._token_flag()} -v $HOME/.cache/huggingface:/hf --entrypoint hf {shlex.quote(self._image(m))} "
               f"download {shlex.quote(m['serve'])} --max-workers 16 "
               + " ".join(f"--exclude {shlex.quote(x)}" for x in m.get("download_exclude", [])))
        threads = [threading.Thread(target=r.run, args=(cmd,), kwargs={"timeout": timeout, "check": False})
                   for r in self.nodes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def start(self, m: dict, timeout: float, log_path: Path) -> None:
        self.stop(m)
        serve = ["vllm", "serve", m["serve"], "--host", "0.0.0.0", "--port", str(self.port),
                 *map(str, m.get("args", []))]
        if not self.workers:
            self._run_container(self.r, m, " ".join(map(shlex.quote, serve)))
        else:
            ips, n = self.mc["node_ips"], len(self.nodes)   # head first, then one per worker
            par = ["--tensor-parallel-size", str(self.mc.get("tp", n)), "--distributed-executor-backend", "mp",
                   "--nnodes", str(n), "--master-addr", ips[0], "--master-port", str(self.mc.get("master_port", 29501))]
            self._run_container(self.r, m, " ".join(map(shlex.quote, serve + par + ["--node-rank", "0"])), ips[0])
            for rank, (w, ip) in enumerate(zip(self.workers, ips[1:]), start=1):
                self._run_container(w, m, " ".join(map(shlex.quote, serve + par +
                                                       ["--node-rank", str(rank), "--headless"])), ip)
        wait_ready(self, timeout, log_path, expect=len(self.nodes))

    def logs(self, m: dict) -> str:
        return "".join(f"===== node{i}\n" + (r.run(f"docker logs --tail 300 {self.NAME} 2>&1", check=False).stdout or "")
                       for i, r in enumerate(self.nodes))

    def stop(self, m: dict) -> None:
        for r in self.nodes:
            r.run(f"docker rm -f {self.NAME} >/dev/null 2>&1", check=False, timeout=300)


class MLXLauncher:
    """Apple silicon: `mlx_lm.server` (run with uvx, so the Mac needs only uv). Per model: `serve` (an MLX repo,
    e.g. mlx-community/...-4bit) and `weights_gb`; a model that would not fit this Mac's memory is skipped."""

    def __init__(self, remote: Remote, mc: dict):
        self.r, self.mc, self.port = remote, mc, mc["port"]
        self.server = mc.get("server", "uvx --from mlx-lm mlx_lm.server")
        self.memory_gb = 0.0

    def setup(self) -> None:
        self.r.run(f"mkdir -p {self.mc['remote_dir']}")

    def running(self) -> list[str]:
        out = self.r.run("pgrep -fl mlx_lm.server", check=False).stdout or ""
        return [ln.strip() for ln in out.splitlines() if ln.strip()]

    def environment(self) -> dict:
        info = {}
        for key, cmd in {"model": "sysctl -n hw.model", "chip": "sysctl -n machdep.cpu.brand_string",
                         "memory_bytes": "sysctl -n hw.memsize", "macos": "sw_vers -productVersion",
                         "mlx_lm": "uvx --from mlx-lm python -c 'import mlx_lm; print(mlx_lm.__version__)'"}.items():
            info[key] = (self.r.run(cmd, check=False).stdout or "").strip().splitlines()[-1:] or [""]
            info[key] = info[key][0]
        if info.get("memory_bytes", "").isdigit():
            self.memory_gb = int(info["memory_bytes"]) / 2**30
        return info

    def fits(self, m: dict) -> bool:
        return not self.memory_gb or not m.get("weights_gb") or ram_needed(m["weights_gb"])[0] <= self.memory_gb + 0.5

    def repair(self, model: dict) -> None:
        pass

    def prefetch(self, m: dict, model: dict, timeout: float) -> None:
        self.r.run(f"uvx --from huggingface_hub hf download {shlex.quote(m['serve'])}", timeout=timeout, check=False)

    def start(self, m: dict, timeout: float, log_path: Path) -> None:
        self.stop(m)
        logf = f"{self.mc['remote_dir']}/mlx-server.log"
        self.r.run(f"nohup {self.server} --model {shlex.quote(m['serve'])} --host 127.0.0.1 --port {self.port} "
                   f"{' '.join(map(shlex.quote, map(str, m.get('args', []))))} > {logf} 2>&1 < /dev/null &",
                   check=False, timeout=60)
        wait_ready(self, timeout, log_path)

    def logs(self, m: dict) -> str:
        return self.r.run(f"tail -n 300 {self.mc['remote_dir']}/mlx-server.log", check=False).stdout or ""

    def stop(self, m: dict) -> None:
        self.r.run("pkill -f mlx_lm.server; sleep 2", check=False, timeout=60)


LAUNCHERS = {"tt": TTLauncher, "vllm-docker": VLLMDockerLauncher, "mlx": MLXLauncher}


def wait_ready(launcher, timeout: float, log_path: Path, expect: int = 1) -> None:
    deadline = time.monotonic() + timeout
    while not launcher.r.dry_run:
        if launcher.r.run(f"curl -fsS -m 5 http://127.0.0.1:{launcher.port}/v1/models", check=False).returncode == 0:
            return
        if len(launcher.running()) < expect:
            log_path.write_text(launcher.logs({}))
            raise RuntimeError(f"server exited during boot; see {log_path}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"not ready after {timeout:.0f}s")
        time.sleep(15)


def ram_needed(weights_gb: float) -> tuple[int, int]:
    """(minimum, recommended) unified memory for an MLX model: macOS lets the GPU use about 75% of memory, which has to
    hold the weights plus the KV cache (about 4 GB for one user, 12 GB for the 32-user and 32K-token tests)."""
    sizes = [16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]
    pick = lambda need: next((s for s in sizes if s * 0.75 >= need), sizes[-1])
    return pick(weights_gb + 4), pick(weights_gb + 12)


# ---------------------------------------------------------------------------------------------------- clients

def served_model_id(r: Remote, port: int, pm: dict) -> str:
    if pm.get("served_name"):
        return pm["served_name"]
    out = r.run(f"curl -fsS http://127.0.0.1:{port}/v1/models", timeout=30).stdout
    ids = [m["id"] for m in json.loads(out)["data"]]
    return pm["serve"] if pm["serve"] in ids else ids[0]   # mlx_lm.server lists every cached model


def bench_cmd(mc: dict, d: dict, model: dict, pm: dict, served: str, isl: int, osl: int, conc: int,
              n: int, out_dir: str, fname: str, meta: dict, prefix: int = 0, seed: int = 0) -> str:
    """`vllm bench serve` on the machine itself (so network latency never enters TTFT), or for `client: builtin`
    tools/loadgen.py, which takes the same flags and writes the same result fields."""
    backend = model.get("backend", d["backend"])
    client = ([mc["bench_python"], "bench", "serve"] if mc.get("client", "vllm") == "vllm" else
              ["uv", "run", "--quiet", "--script", f"{mc['remote_dir']}/loadgen.py"])
    args = client + [
        "--backend", backend, "--base-url", f"http://127.0.0.1:{mc['port']}",
        "--model", served, "--tokenizer", pm.get("tokenizer", model["hf_id"]),
        "--dataset-name", "random", "--random-input-len", str(isl - prefix), "--random-output-len", str(osl),
        "--random-range-ratio", "0", "--ignore-eos",
        "--num-prompts", str(n), "--max-concurrency", str(conc), "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
        "--save-result", "--result-dir", out_dir, "--result-filename", fname, "--seed", str(seed),
    ]
    if prefix:   # the first `prefix` tokens of every prompt are the same (a resent agent context)
        args += ["--random-prefix-len", str(prefix)]
    temp = model.get("temperature", d.get("temperature"))
    if temp is not None:
        args += ["--temperature", str(temp)]
    if meta:
        args += ["--metadata", *[f"{k}={v}" for k, v in meta.items()]]
    if backend == "openai-chat":
        args += ["--endpoint", "/v1/chat/completions"]
    if mc.get("client") == "builtin":     # fallback tokenizer: the served repo (gated originals need a token)
        args += ["--tokenizer", pm["serve"]]
    return " ".join(map(shlex.quote, args))


# ---------------------------------------------------------------------------------------------------- run

def cmd_run(cfg: dict, a: argparse.Namespace) -> int:
    if a.machine not in cfg["machines"]:
        sys.exit(f"unknown machine {a.machine!r}; known: {', '.join(cfg['machines'])}")
    mc, d = cfg["machines"][a.machine], cfg["defaults"]
    if "ssh" not in mc:
        sys.exit(f"machines.yaml has no ssh target for {a.machine!r} (see machines.example.yaml)")
    r = Remote(mc["ssh"], a.dry_run)
    # paths get shell-quoted, so expand ~ to the remote $HOME up front
    home = "~" if a.dry_run else r.run("echo $HOME").stdout.strip().splitlines()[-1]
    mc = {k: v.replace("~", home, 1) if isinstance(v, str) and v.startswith("~") else v for k, v in mc.items()}
    launcher = LAUNCHERS[mc["launcher"]](r, mc)

    models = [(m, pm) for m, pm in machine_models(cfg, a.machine, a.models_from)
              if (m["id"] in a.model if a.model else m.get("enabled", True))]
    workloads = [w for w in cfg["workloads"] if (w["name"] in a.workload if a.workload else w.get("enabled", True))]
    if not models or not workloads:
        sys.exit("nothing selected")
    busy = [] if a.dry_run else launcher.running()
    if busy:
        sys.exit(f"{a.machine} is already serving {busy}; stop it first")

    launcher.setup()
    if mc.get("client") == "builtin":
        r.push(ROOT / "tools" / "loadgen.py", mc["remote_dir"])
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    env_info = {} if a.dry_run else scrub(launcher.environment())
    failures = 0
    for model, pm in models:
        if hasattr(launcher, "fits") and not launcher.fits(pm):
            log(f"== {model['id']}: skipped, needs {ram_needed(pm['weights_gb'])[0]} GB of unified memory")
            continue
        if a.max_model_len and "args" in pm:   # a longer context than the model's usual config (the 128K tests)
            args = list(pm["args"])
            if "--max-model-len" in args:
                args[args.index("--max-model-len") + 1] = str(a.max_model_len)
            else:
                args += ["--max-model-len", str(a.max_model_len)]
            pm = {**pm, "args": args}
        args = pm.get("args", [])
        ctx = int(args[args.index("--max-model-len") + 1]) if "--max-model-len" in args else None
        local = (Path("/tmp/bench-dry-run") if a.dry_run else RESULTS / a.machine) / model["id"] / stamp
        local.mkdir(parents=True, exist_ok=True)
        remote_out = f"{mc['remote_dir']}/results/{model['id']}/{stamp}"
        (local / "meta.json").write_text(json.dumps({
            "machine": a.machine, "hardware": mc["hardware"], "model": model["id"], "hf_id": model["hf_id"],
            "serve": pm, "started": stamp, "environment": env_info}, indent=2))

        log(f"== {model['id']}: starting {pm['serve']} on {a.machine}")
        t0 = time.monotonic()
        try:
            launcher.prefetch(pm, model, d["serve_timeout_s"])
            try:
                launcher.start(pm, d["serve_timeout_s"], local / "serve.log")
            except RuntimeError as e:   # e.g. a device left dirty by the previous model: repair, retry once
                log(f"   !! start failed ({e}); retrying once")
                launcher.stop(pm)
                launcher.repair(model)
                launcher.start(pm, d["serve_timeout_s"], local / "serve-retry.log")
            boot_s = time.monotonic() - t0
            served = "DRY" if a.dry_run else served_model_id(r, mc["port"], pm)
            log(f"   ready in {boot_s:.0f}s as {served!r}")
            r.run(f"mkdir -p {remote_out}")

            maxc = pm.get("max_concurrency", model.get("max_concurrency"))   # per machine wins (null = no cap)
            if d.get("warmup_prompts"):
                log("   warm-up")
                r.run(bench_cmd(mc, d, model, pm, served, 128, 32, 1, d["warmup_prompts"], remote_out,
                                "warmup.json", {}), timeout=d["request_timeout_s"], check=False)
            # per-model overrides: `workloads` (names to run), `prompts_per_concurrency`, `min_prompts`
            for w in [w for w in workloads if w["name"] in model.get("workloads", [w["name"]])]:
                if ctx and w["isl"] + w["osl"] > ctx:
                    log(f"   {w['name']}: skipped, needs {w['isl'] + w['osl']} tokens of context (served with {ctx}; "
                        "run with --max-model-len)")
                    continue
                if w.get("warmup"):   # one unmeasured request at this length, with its own prompt
                    log(f"   {w['name']}: warm-up at {w['isl']} tokens")
                    r.run(bench_cmd(mc, d, model, pm, served, w["isl"], 16, 1, 1, remote_out, "prime.json", {},
                                    seed=zlib.crc32(f"warmup-{w['name']}".encode())),
                          timeout=d["request_timeout_s"], check=False)
                for conc in w["concurrency"]:
                    if (maxc and conc > maxc) or (a.users and conc not in a.users):
                        continue
                    per = model.get("prompts_per_concurrency", w.get("prompts_per_concurrency", d["prompts_per_concurrency"]))
                    n = max(model.get("min_prompts", d["min_prompts"]), per * conc)
                    # every run gets its own prompts: with one seed, each run would resend the previous run's
                    # prompts and a server with prefix caching would answer those from cache
                    seed = zlib.crc32(f"{w['name']}-{w['isl']}-{conc}".encode())
                    if w.get("prefix"):
                        # the shared prefix depends on the seed and the prompt count, so prime with both: n requests
                        # of the prefix plus 16 tokens (the first one fills the cache), unmeasured
                        log(f"   {w['name']} c={conc}: priming the shared {w['prefix']}-token prefix")
                        r.run(bench_cmd(mc, d, model, pm, served, w["prefix"] + 16, 16, 1, n, remote_out, "prime.json",
                                        {}, w["prefix"], seed), timeout=d["request_timeout_s"] * n, check=False)
                    fname = f"{w['name']}_isl{w['isl']}_osl{w['osl']}_c{conc}.json"
                    meta = {"machine": a.machine, "hardware": mc["hardware"], "model_id": model["id"],
                            "workload": w["name"], "serve": pm["serve"], "boot_s": round(boot_s), "seed": seed,
                            **({"max_model_len": ctx} if ctx else {})}
                    log(f"   {w['name']} isl={w['isl']} osl={w['osl']} c={conc} n={n}")
                    with PowerMeter([] if a.dry_run else mc.get("power_plugs", [])) as pmeter:
                        res = r.run(bench_cmd(mc, d, model, pm, served, w["isl"], w["osl"], conc, n,
                                              remote_out, fname, meta, w.get("prefix", 0), seed), timeout=d["request_timeout_s"] * n, check=False)
                    pmeter.save(local / fname.replace(".json", ".power.json"))
                    (local / fname.replace(".json", ".log")).write_text(res.stdout or "")
                    if res.returncode != 0:
                        failures += 1
                        log(f"   !! bench failed ({res.returncode}); continuing")
                    if not a.dry_run and r.run(f"curl -fsS -m 30 http://127.0.0.1:{mc['port']}/v1/models",
                                               check=False).returncode != 0:
                        raise RuntimeError(f"server stopped answering during {fname}")
        except Exception as e:  # keep going with the next model
            failures += 1
            log(f"   !! {model['id']} failed: {e}")
        finally:
            r.pull(remote_out, local)
            if not a.dry_run:
                (local / "server.log").write_text(launcher.logs(pm))
            log(f"   stopping {pm['serve']}")
            launcher.stop(pm)
    log(f"done: {failures} failure(s); results under {RESULTS / a.machine}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------------------------------- report

def load_results(latest: bool) -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("*/*/*/*.json")):
        if f.name in ("meta.json", "warmup.json", "prime.json") or f.name.endswith(".power.json"):
            continue
        j = json.loads(f.read_text())
        if f.parts[-2].startswith("failed") or "mean_ttft_ms" not in j or not j.get("completed"):
            continue                                        # crashed runs, or every request failed
        fm = re.search(r"_isl(\d+)_osl(\d+)_c(\d+)$", f.stem)    # <workload>_isl<N>_osl<N>_c<N>.json
        isl, osl, conc = (int(x) for x in fm.groups()) if fm else (0, 0, j.get("max_concurrency") or 0)
        ttft, tpot = j["mean_ttft_ms"], j["mean_tpot_ms"]
        watts = wall_watts(j, f.with_name(f.stem + ".power.json"))
        rows.append({
            "machine": j.get("machine", f.parts[-4]), "model": j.get("model_id", f.parts[-3]),
            "run": f.parts[-2], "workload": j.get("workload", f.stem.split("_")[0]),
            "isl": isl, "osl": osl, "concurrency": conc,
            "completed": j.get("completed"), "failed": j.get("failed", 0),
            "ttft_ms_mean": round(ttft, 1), "ttft_ms_p90": round(j.get("p90_ttft_ms", 0), 1),
            "prefill_tok_s": round(isl / (ttft / 1000), 1) if ttft else None,
            "tpot_ms_mean": round(tpot, 2),
            "decode_tok_s_user": round(1000 / tpot, 2) if tpot else None,
            "output_tok_s_total": round(j.get("output_throughput", 0), 2),
            "e2e_ms_mean": round(j.get("mean_e2el_ms", 0), 1) or None,
            "req_s": round(j.get("request_throughput", 0), 4) or None,
            "boot_s": j.get("boot_s"),
            "wall_w": watts,
            "out_tok_per_j": round(j.get("output_throughput", 0) / watts, 3) if watts else None,
        })
    if latest:  # keep only the newest result per machine/model/workload/concurrency
        key = lambda r: (r["machine"], r["model"], r["workload"], r["isl"], r["osl"], r["concurrency"])
        newest: dict[tuple, str] = {}
        for r in rows:
            newest[key(r)] = max(newest.get(key(r), ""), r["run"])
        rows = [r for r in rows if r["run"] == newest[key(r)]]
    rows.sort(key=lambda r: (r["model"], r["isl"], r["osl"], r["concurrency"], r["machine"]))
    return rows


def cmd_report(cfg: dict, a: argparse.Namespace) -> int:
    rows = load_results(a.latest)
    if not rows:
        sys.exit("no results yet")
    with open(RESULTS / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    show = ["machine", "model", "workload", "isl", "osl", "concurrency", "ttft_ms_mean", "prefill_tok_s",
            "decode_tok_s_user", "output_tok_s_total", "wall_w", "out_tok_per_j", "failed"]
    md = ["| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    md += ["| " + " | ".join(str(r[c]) for c in show) + " |" for r in rows]
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# ---------------------------------------------------------------------------------------------------- site

SITE_RESULTS, SITE_QUALITY, SITE_TALOS = "results.csv", "quality.csv", "talos.csv"


def cmd_site(cfg: dict, a: argparse.Namespace) -> int:
    """docs/: the page (site/index.html in a small stand-in for the claude.ai dashboard runtime) and its data."""
    models = {m["id"]: m for m in cfg["models"]}
    results = load_results(latest=True)
    out = []
    for r in results:
        m, mc = models.get(r["model"]), cfg["machines"].get(r["machine"], {})
        if not m or m.get("chart") is False or mc.get("talos_of"):
            continue
        base = models.get(m.get("same_as"), m)               # a second build of a model charts on that model's row
        pm = m["per_machine"].get(mc.get("models_from", r["machine"]), {})
        out.append({
            "system": mc.get("label", r["machine"]) + (f" ({m['build']})" if m.get("build") else ""),
            "family": mc.get("family", ""),
            "model": base["id"], "arch": base["arch"], "weights": pm.get("weights", base.get("weights", "")),
            "total_params_b": base["params_b"]["total"], "active_params_b": base["params_b"]["active"],
            "workload": r["workload"], "isl": r["isl"], "osl": r["osl"], "users": r["concurrency"],
            "decode_tok_s_user": r["decode_tok_s_user"], "total_tok_s": r["output_tok_s_total"],
            "ttft_ms": r["ttft_ms_mean"], "e2e_ms": r["e2e_ms_mean"], "req_s": r["req_s"],
            "wall_w": r["wall_w"], "tok_per_j": r["out_tok_per_j"],
        })
    quality = [{"model": m["id"], "aa_index": q["aa_index"], "aa_variant": q["variant"],
                "aa_estimate": "yes" if q.get("estimate") else "no", "index_version": q["version"],
                "source_url": q["url"]} for m in cfg["models"] if (q := m.get("quality"))]
    DOCS.mkdir(exist_ok=True)
    for name, rows in ((SITE_RESULTS, out), (SITE_QUALITY, quality)):
        with open(DOCS / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    page = (ROOT / "site" / "index.html").read_text()
    if not any(o["workload"] == "power" and o["users"] == 32 and o["tok_per_j"] for o in out):
        page = re.sub(r'\s*<button type="button" data-users="32"[^>]*>32 users</button>', "", page)
    aa = "".join(f'<li><a href="{html.escape(q["source_url"])}">{html.escape(q["model"])}</a>: {q["aa_index"]} '
                 f'({html.escape(q["aa_variant"])}{", estimate" if q["aa_estimate"] == "yes" else ""})</li>'
                 for q in quality)
    systems = ", ".join(sorted({o["system"] for o in out}))
    page = page.replace("__QUALITY__", aa).replace("__SYSTEMS__", html.escape(systems))
    write_page("index.html", page, {"results": out, "aa_index": quality}, "Tokens per Joule",
               "Output tokens per joule of wall power for open-weight models on Tenstorrent, NVIDIA, Apple and AMD "
               "hardware, with speed, running cost and published quality scores.")
    (DOCS / ".nojekyll").touch()
    print(f"wrote {DOCS / 'index.html'} ({len(out)} result rows, {len(quality)} quality scores)")

    pairs = talos_pairs(cfg, results)
    with open(DOCS / SITE_TALOS, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    write_page("talos.html", (ROOT / "site" / "talos.html").read_text(), {"pairs": pairs}, "Talos Linux vs Ubuntu",
               "The same QuietBox 2 and DGX Spark on Talos Linux and on Ubuntu: the change in throughput, per-user "
               "speed and time to first token for each model and test.")
    print(f"wrote {DOCS / 'talos.html'} ({len(pairs)} paired tests)")
    return 0


def talos_pairs(cfg: dict, results: list[dict]) -> list[dict]:
    """Each Talos result next to the Ubuntu result for the same machine, model, test and user count."""
    models = {m["id"]: m for m in cfg["models"]}
    key = lambda r, machine: (machine, r["model"], r["workload"], r["isl"], r["osl"], r["concurrency"])
    ubuntu = {key(r, r["machine"]): r for r in results}
    pairs = []
    for t in results:
        mc = cfg["machines"].get(t["machine"], {})
        if not mc.get("talos_of") or not (u := ubuntu.get(key(t, mc["talos_of"]))) or not (m := models.get(t["model"])):
            continue
        base = models.get(m.get("same_as"), m)
        pairs.append({
            "system": mc["label"], "family": mc.get("family", ""),
            "model": base["id"] + (f" ({m['build']})" if m.get("build") else ""),
            "workload": t["workload"], "isl": t["isl"], "osl": t["osl"], "users": t["concurrency"],
            "ubuntu_tok_s": u["output_tok_s_total"], "talos_tok_s": t["output_tok_s_total"],
            "ubuntu_ttft_ms": u["ttft_ms_mean"], "talos_ttft_ms": t["ttft_ms_mean"],
            "ubuntu_decode_tok_s": u["decode_tok_s_user"], "talos_decode_tok_s": t["decode_tok_s_user"],
        })
    return pairs


def write_page(name: str, page: str, data: dict[str, list[dict]], title: str, desc: str) -> None:
    """docs/<name>: a page from site/ inside site/shell.html, with its data inlined."""
    # each dashboard script runs in its own scope
    page = re.sub(r"<script>(.*?)</script>", lambda x: f"<script>\n(() => {{{x.group(1)}}})();\n</script>", page, flags=re.S)
    data = {k: [{c: ("" if v is None else str(v)) for c, v in o.items()} for o in rows] for k, rows in data.items()}
    doc = ((ROOT / "site" / "shell.html").read_text()
           .replace("__TITLE__", html.escape(title)).replace("__DESC__", html.escape(desc))
           .replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__PAGE__", page)
           .replace("__DATE__", f"{dt.date.today():%Y-%m-%d}"))
    (DOCS / name).write_text(doc)


def cmd_ram(cfg: dict, a: argparse.Namespace) -> int:
    print("| model | MLX build | weights | minimum memory | for the full test |\n|---|---|---|---|---|")
    for m in cfg["models"]:
        pm = m.get("per_machine", {}).get("mac")
        if pm and pm.get("weights_gb"):
            lo, hi = ram_needed(pm["weights_gb"])
            print(f"| {m['id']} | `{pm['serve']}` | {pm['weights_gb']} GB | {lo} GB | {hi} GB |")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "bench.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="serve and benchmark models on one machine")
    pr.add_argument("machine")
    pr.add_argument("-m", "--model", action="append", help="model id (repeatable); default: all enabled")
    pr.add_argument("-w", "--workload", action="append", help="workload name (repeatable); default: all enabled")
    pr.add_argument("-u", "--users", action="append", type=int, help="only these user counts (repeatable)")
    pr.add_argument("--max-model-len", type=int, help="serve vLLM models with this much context (the 128K tests)")
    pr.add_argument("--models-from", help="serve another machine's model builds (e.g. spark's, on spark-pair)")
    pr.add_argument("--dry-run", action="store_true")
    rp = sub.add_parser("report", help="results/summary.{csv,md}")
    rp.add_argument("--latest", action="store_true", help="only the newest run of each test")
    sub.add_parser("site", help="build docs/ (the published page)")
    sub.add_parser("ram", help="unified memory each model needs on a Mac")
    a = p.parse_args()
    cfg = load_config(Path(a.config))
    return {"run": cmd_run, "report": cmd_report, "site": cmd_site, "ram": cmd_ram}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
