#!/usr/bin/env python3
"""Optional: sample every plug in machines.yaml once a second, continuously, into results/power/<plug>.csv
(`epoch,watts,volts,amps,wh_total`). bench.py already records power during each test; this is for watching idle
draw or anything run outside bench.py.

    nohup tools/powerlog.py > results/power/powerlog.log 2>&1 &
"""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "power"


def plugs() -> list[str]:
    text = (ROOT / "machines.yaml").read_text()
    found = []
    for m in re.finditer(r"power_plugs:\s*\[([^\]]*)\]", text):
        found += [p.strip().strip("'\"") for p in m.group(1).split(",") if p.strip()]
    return sorted(set(found))


def sample(plug: str) -> str | None:
    try:
        with urllib.request.urlopen(f"http://{plug}/rpc/Switch.GetStatus?id=0", timeout=2) as r:
            d = json.load(r)
        return f"{time.time():.1f},{d['apower']},{d['voltage']},{d['current']},{d['aenergy']['total']}\n"
    except Exception as e:
        print(f"{time.strftime('%H:%M:%S')} {plug}: {e}", file=sys.stderr, flush=True)
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ps = plugs()
    print(f"sampling {ps}", flush=True)
    files = {p: open(OUT / f"{p}.csv", "a", buffering=1) for p in ps}
    with ThreadPoolExecutor(len(ps)) as pool:
        while True:
            t = time.monotonic()
            for p, line in zip(ps, pool.map(sample, ps)):
                if line:
                    files[p].write(line)
            time.sleep(max(0.0, 1.0 - (time.monotonic() - t)))


if __name__ == "__main__":
    main()
