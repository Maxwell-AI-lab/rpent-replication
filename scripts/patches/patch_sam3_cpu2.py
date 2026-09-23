from pathlib import Path
p = Path("/data/rpent/rpent/rpent/robots/components/sam3_server.py")
t = p.read_text()
if "cpu-mode tensor shim" in t:
    print("already"); raise SystemExit
SHIM = '''
# --- cpu-mode tensor shim (auto-patched): redirect any cuda allocation to cpu ---
if os.environ.get("SAM3_DEVICE", "").lower() == "cpu":
    import torch as _t
    def _cpu_dev(d):
        if isinstance(d, str) and d.startswith("cuda"): return "cpu"
        if d is not None and hasattr(d, "type") and d.type == "cuda": return "cpu"
        return d
    for _n in ("zeros", "ones", "empty", "full", "tensor", "arange", "randn", "rand", "eye", "linspace"):
        _f = getattr(_t, _n)
        def _mk(f=_f):
            def g(*a, **k):
                if "device" in k: k["device"] = _cpu_dev(k["device"])
                return f(*a, **k)
            return g
        setattr(_t, _n, _mk())
    _t.Tensor.cuda = lambda self, *a, **k: self
    _to_orig = _t.Tensor.to
    def _to2(self, *a, **k):
        a = tuple(_cpu_dev(x) for x in a)
        if "device" in k: k["device"] = _cpu_dev(k["device"])
        return _to_orig(self, *a, **k)
    _t.Tensor.to = _to2
    _dev_orig = _t.device
# --- end cpu-mode tensor shim ---
'''
anchor = "# --- end NPU compat ---"
assert anchor in t, "NPU block anchor missing"
t = t.replace(anchor, anchor + "\n" + SHIM, 1)
import ast; ast.parse(t)
p.write_text(t)
print("shim installed + syntax ok")
