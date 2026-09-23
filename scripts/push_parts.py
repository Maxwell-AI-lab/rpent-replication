#!/usr/bin/env python3
"""Parallel chunk pusher: Mac -> 155 via scp (through SSRDOG ProxyCommand)."""
import subprocess, sys, time, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path("/Users/max/code/ai/embody/rpent-repro/downloads/pi05_parts")
DST = "rpent155:/data/rpent/checkpoints/pi05_parts"
CONC = int(os.environ.get("CONC", "24"))

def push(f: Path) -> bool:
    for attempt in range(6):
        r = subprocess.run(
            ["scp", "-q", "-o", "Compression=no", "-o", "ConnectTimeout=20",
             str(f), f"{DST}/{f.name}"],
            capture_output=True, timeout=3600)
        if r.returncode == 0:
            print(f"OK {f.name} att{attempt}", flush=True)
            return True
        print(f"retry {f.name} att{attempt} rc={r.returncode} {r.stderr.decode()[:120]}", flush=True)
        time.sleep(3)
    print(f"FAILED {f.name}", flush=True)
    return False

files = sorted(SRC.glob("p*"))
print(f"pushing {len(files)} chunks, concurrency={CONC}", flush=True)
t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    results = list(ex.map(push, files))
ok = sum(results)
print(f"DONE {ok}/{len(files)} in {(time.time()-t0)/60:.1f} min", flush=True)
sys.exit(0 if ok == len(files) else 1)
