import re
from pathlib import Path
root = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/sam3")
# inference-path files only (skip train/)
files = ["model/position_encoding.py", "model/io_utils.py",
         "model/video_tracking_multiplex.py", "model/model_misc.py"]
GUARD = '_DEV = "cuda" if __import__("torch").cuda.is_available() else "cpu"'
for rel in files:
    p = root / rel
    if not p.exists():
        print("missing", rel); continue
    t = p.read_text()
    if "_DEV =" in t:
        print("already", rel); continue
    t = t.replace('device="cuda"', 'device=_DEV')
    t = t.replace('.cuda()', '.to(_DEV)')
    lines = t.splitlines(keepends=True)
    # insert guard after last top-level import
    last_import = 0
    for i, l in enumerate(lines[:60]):
        if l.startswith(("import ", "from ")) and not l.startswith("    "):
            last_import = i
    lines.insert(last_import + 1, GUARD + "\n")
    t = "".join(lines)
    import ast; ast.parse(t)
    p.write_text(t)
    print("patched", rel)
