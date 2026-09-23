from pathlib import Path
import ast
root = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/sam3/model")
GUARD = '_DEV = "cuda" if __import__("torch").cuda.is_available() else "cpu"'
files = ["sam3_image.py", "sam3_base_predictor.py", "sam3_video_base.py",
         "video_tracking_multiplex.py", "sam3_video_inference.py"]
for rel in files:
    p = root / rel
    if not p.exists():
        print("missing", rel); continue
    t = p.read_text()
    changed = False
    if "_DEV =" not in t:
        lines = t.splitlines(keepends=True)
        last = 0
        for i, l in enumerate(lines[:60]):
            if l.startswith(("import ", "from ")) and not l.startswith("    "):
                last = i
        lines.insert(last + 1, GUARD + "\n")
        t = "".join(lines); changed = True
    if ".to(torch.bfloat16)" in t:
        t = t.replace(".to(torch.bfloat16)",
                      '.to(torch.bfloat16 if _DEV != "cpu" else torch.float32)')
        changed = True
    if not changed:
        print("nochange", rel); continue
    ast.parse(t)
    p.write_text(t)
    print("patched", rel)
