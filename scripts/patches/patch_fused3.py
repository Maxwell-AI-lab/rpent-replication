from pathlib import Path
import ast
p = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/sam3/perflib/fused.py")
t = p.read_text()
old = '    if _os.environ.get("SAM3_DEVICE", "").lower() == "cpu":'
new = '    if _os.environ.get("SAM3_FUSED", "0") != "1":  # default: portable path (CPU & NPU)'
assert old in t
t = t.replace(old, new, 1)
ast.parse(t)
p.write_text(t)
print("universal fallback on")
