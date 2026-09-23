#!/usr/bin/env python3
"""Patch rpent GPU servers for Ascend NPU via torch_npu transfer_to_npu.

Inserts a guarded compatibility block at module level of:
  - rpent/robots/components/pi05_vla_server.py
  - rpent/robots/components/sam3_server.py
On CUDA boxes the block is a no-op (torch.cuda available -> no transfer).
Idempotent: skips if marker already present.
"""
import sys
from pathlib import Path

BLOCK = '''\
# --- NPU compat (auto-patched; no-op on CUDA boxes) ---
try:
    import torch_npu  # noqa: F401
    import torch
    if not torch.cuda.is_available():
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
except Exception:
    pass
# --- end NPU compat ---
'''

MARKER = "NPU compat (auto-patched"

def patch(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        print(f"already patched: {path}")
        return
    lines = text.splitlines(keepends=True)
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.strip() == "import torch" and not line.startswith(" "):
            out.append("\n" + BLOCK)
            inserted = True
    if not inserted:
        # sam3_server imports torch lazily; append at module level after imports
        for i, line in enumerate(out):
            if line.startswith("logger = get_logger("):
                out.insert(i, BLOCK + "\n")
                inserted = True
                break
    if not inserted:
        print(f"WARN: no anchor found in {path}", file=sys.stderr)
        return
    path.write_text("".join(out))
    print(f"patched: {path}")

if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    for rel in ("rpent/robots/components/pi05_vla_server.py",
                "rpent/robots/components/sam3_server.py"):
        patch(base / rel)
