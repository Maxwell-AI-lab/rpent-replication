from pathlib import Path
import ast
p = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/sam3/perflib/fused.py")
t = p.read_text()
if "cpu fallback" in t:
    print("already"); raise SystemExit
old = """def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
"""
new = """def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    import os as _os
    if _os.environ.get("SAM3_DEVICE", "").lower() == "cpu":
        # cpu fallback: plain fp32 linear + activation (no fused bf16 kernel)
        w = linear.weight.detach().to(mat1.dtype)
        b = linear.bias.detach().to(mat1.dtype)
        flat = mat1.reshape(-1, mat1.shape[-1])
        y = torch.nn.functional.linear(flat, w, b)
        y = activation(y)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
"""
assert old in t
t = t.replace(old, new, 1)
ast.parse(t)
p.write_text(t)
print("fused.py patched")
