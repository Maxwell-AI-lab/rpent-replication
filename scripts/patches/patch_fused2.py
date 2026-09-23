from pathlib import Path
import ast
p = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/sam3/perflib/fused.py")
t = p.read_text()
old = """        y = torch.nn.functional.linear(flat, w, b)
        y = activation(y)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))"""
new = """        y = torch.nn.functional.linear(flat, w, b)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            y = torch.nn.functional.gelu(y)
        else:
            y = torch.nn.functional.relu(y)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))"""
assert old in t
t = t.replace(old, new, 1)
ast.parse(t)
p.write_text(t)
print("fixed")
